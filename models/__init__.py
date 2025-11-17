
"""
Centralized model loading with a shared artifact manager.

This refactor makes load_model/load_manipulated_model thin wrappers:
 - They parse the "<dataset>_<arch>" string (with legacy alias support)
 - They look up dataset and architecture specs
 - They delegate loading to a shared ArtifactManager that:
     1) checks local checkpoints
     2) optionally tries torch.hub/HuggingFace fetchers
     3) falls back to an architecture-defined training function if provided

Existing specialized loader functions (e.g., load_resnet20_model_normal) are
retained and reused by the ArtifactManager through ModelSpec definitions.
"""

# System
import sys

import torch
sys.path.append('pytorch_resnet_cifar10/')

import os

# Libs
import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# Own sources
import models.resnet_freeze_bn as resnet_freeze_bn
import models.resnet_nbn as resnet_nbn
#for drop out use the one from below
import models.resnet_dropout as resnet_dropout
import models.resnet as resnet_normal
from models.resnet50 import (
    load_imagenet_resnet18_model,
    load_imagenet_resnet18xbn_model,
    load_imagenet_resnet18_model_local,
    load_imagenet_resnet18xbn_model_local,
    load_imagenet_resnet18_manipulated,
    load_imagenet_resnet18xbn_manipulated,
    load_imagenet_resnet50_model,
    load_imagenet_resnet50_model_local,
    load_imagenet_resnet50_manipulated,
)
import models.vgg as vgg
from models.mobilenet_v3_small import mobilenet_v3_small, MobileNetV3Small, transfer_from_torchvision_mnv3_small
from models.vit_b_16 import vit_b_16, ViTB16, transfer_from_torchvision_vit
from models.vit_b_16bn import (
    vit_b_16_bn,
    ViTB16BN,
    transfer_from_torchvision_vit_bn,
    load_vit_b_16bn_model_normal,
    load_vit_b_16bn_model_manipulated,
)
from utils.config import DatasetEnum
from utils.train_config import (
    get_train_config,
    get_warmup_config,
    is_auto_train_enabled,
    is_warmup_enabled,
)
import torch.nn as nn
from plot import replace_bn


# Dataset metadata used for generic model loaders. Maps dataset key (lowercase)
# to (DatasetEnum, num_classes).
_DATASET_METADATA = {
    'cifar10': (DatasetEnum.CIFAR10, 10),
    'cifar100': (DatasetEnum.CIFAR100, 100),
    'gtsrb': (DatasetEnum.GTSRB, 43),
    'imagenet': (DatasetEnum.IMAGENET, 1000),
}


# --------------------------------------------------------------------------------------
# Spec types and artifact manager
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    key: str
    enum: DatasetEnum
    num_classes: int


@dataclass(frozen=True)
class ModelSpec:
    """Describes how to load or construct a given architecture.

    Fields:
      arch_key: canonical architecture key, e.g. "resnet20", "vgg13bn".
      # path_fn returns (dir_path, file_name) for a given dataset key and index.
      path_fn: (dataset_key: str, index: int) -> Tuple[str, str]
      # clean/manipulated loaders consume the resolved checkpoint path.
      load_clean: Callable[[str, torch.device, DatasetEnum, int], torch.nn.Module]
      load_manipulated: Callable[[str, torch.device, DatasetEnum, int], torch.nn.Module]
      # Optional recovery hooks used by ArtifactManager if a checkpoint is missing.
      hub_fetcher: Optional[Callable[[torch.device, int, DatasetEnum, str], Optional[Dict[str, Any]]]]
      train_fallback: Optional[Callable[[str, torch.device, int, DatasetEnum], Dict[str, Any]]]
    """
    arch_key: str
    path_fn: Callable[[str, int], Tuple[str, str]]
    load_clean: Callable[[str, torch.device, DatasetEnum, int], torch.nn.Module]
    load_manipulated: Callable[[str, torch.device, DatasetEnum, int], torch.nn.Module]
    hub_fetcher: Optional[Callable[[torch.device, int, DatasetEnum, str], Optional[Dict[str, Any]]]] = None
    train_fallback: Optional[Callable[[str, torch.device, int, DatasetEnum], Dict[str, Any]]] = None


class ArtifactManager:
    """Shared manager that resolves local checkpoints, optional remote fetch, and training fallback."""

    def __init__(self, dataset_specs: Dict[str, DatasetSpec], arch_specs: Dict[str, ModelSpec]):
        self.dataset_specs = dataset_specs
        self.arch_specs = arch_specs

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _prefer_existing_checkpoint(path: str) -> str:
        """Return path if exists; otherwise try swapping extension between .pth and .th; else return original."""
        if os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        alt = stem + ('.th' if ext == '.pth' else '.pth')
        if os.path.exists(alt):
            return alt
        return path

    def _resolve_local_path(self, spec: ModelSpec, dataset_key: str, index: int) -> str:
        dir_path, file_name = spec.path_fn(dataset_key, index)
        self._ensure_dir(dir_path)
        return self._prefer_existing_checkpoint(os.path.join(dir_path, file_name))

    def _recover_checkpoint_if_needed(
        self,
        resolved_path: str,
        spec: ModelSpec,
        device: torch.device,
        num_classes: int,
        dataset: DatasetEnum,
        dataset_key: str,
    ) -> str:
        """Try hub/HF fetcher first; otherwise training fallback. Save to resolved_path and return it."""
        if os.path.exists(resolved_path):
            return resolved_path
        # Try hub fetcher if provided
        if spec.hub_fetcher is not None:
            try:
                fetched = spec.hub_fetcher(device, num_classes, dataset, dataset_key)
                if fetched is not None:
                    torch.save(fetched, resolved_path)
                    return resolved_path
            except Exception as exc:  # noqa: BLE001
                if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
                    print(f"[artifact-manager] hub fetch failed for {dataset_key}:{spec.arch_key} -> {exc}")
        # Train fallback if provided
        if spec.train_fallback is not None:
            try:
                trained = spec.train_fallback(resolved_path, device, num_classes, dataset)
                # train_fallback is expected to save already; we still ensure it exists
                if not os.path.exists(resolved_path):
                    torch.save(trained, resolved_path)
                return resolved_path
            except Exception as exc:  # noqa: BLE001
                if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
                    print(f"[artifact-manager] training fallback failed for {dataset_key}:{spec.arch_key} -> {exc}")
        # If neither worked, just return the unresolved path and let the downstream loader handle its own recovery
        return resolved_path

    def load_clean(self, dataset_key: str, arch_key: str, index: int, device: torch.device) -> torch.nn.Module:
        dataset_key_l = dataset_key.lower()
        if dataset_key_l not in self.dataset_specs:
            raise Exception(f"Unsupported dataset '{dataset_key}'.")
        if arch_key not in self.arch_specs:
            raise Exception(f"Unknown architecture '{arch_key}'.")
        dspec = self.dataset_specs[dataset_key_l]
        aspec = self.arch_specs[arch_key]
        resolved = self._resolve_local_path(aspec, dspec.key, index)
        # Optionally recover to local if missing
        resolved = self._recover_checkpoint_if_needed(resolved, aspec, device, dspec.num_classes, dspec.enum, dspec.key)
        model = aspec.load_clean(resolved, device, dspec.enum, dspec.num_classes)
        return _maybe_run_warmup(model, dspec.enum, device, dspec.num_classes, dspec.key, aspec.arch_key)

    def load_manipulated(self, model_root: str, dataset_key: str, arch_key: str, device: torch.device) -> torch.nn.Module:
        dataset_key_l = dataset_key.lower()
        if dataset_key_l not in self.dataset_specs:
            raise Exception(f"Unsupported dataset '{dataset_key}'.")
        if arch_key not in self.arch_specs:
            raise Exception(f"Unknown architecture '{arch_key}'.")
        dspec = self.dataset_specs[dataset_key_l]
        aspec = self.arch_specs[arch_key]
        # Construct best-effort path inside model_root
        # Prefer model.pth -> model.th, else fallback to model_root as a file path
        if os.path.isdir(model_root):
            cand = [os.path.join(model_root, 'model.pth'), os.path.join(model_root, 'model.th')]
            path = None
            for c in cand:
                if os.path.exists(c):
                    path = c
                    break
            if path is None:
                # If directory is given but no known file exists, assume default name
                path = os.path.join(model_root, 'model.pth')
        else:
            path = model_root
        path = self._prefer_existing_checkpoint(path)
        model = aspec.load_manipulated(path, device, dspec.enum, dspec.num_classes)
        return _maybe_run_warmup(model, dspec.enum, device, dspec.num_classes, dspec.key, aspec.arch_key)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def remove_batchnorm(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, Identity())
        else:
            remove_batchnorm(child)


def _infer_dataset_from_path(path: str) -> DatasetEnum:
    lower = path.lower()
    if 'gtsrb' in lower:
        return DatasetEnum.GTSRB
    if 'cifar100' in lower:
        return DatasetEnum.CIFAR100
    if 'imagenet' in lower:
        return DatasetEnum.IMAGENET
    return DatasetEnum.CIFAR10


def _default_num_classes(dataset: DatasetEnum) -> int:
    if dataset == DatasetEnum.CIFAR100:
        return 100
    if dataset == DatasetEnum.GTSRB:
        return 43
    if dataset == DatasetEnum.IMAGENET:
        return 1000
    return 10


def _load_resnet20_variant_state(
    path: str,
    device: torch.device,
    *,
    dataset_enum: DatasetEnum,
    num_classes: int,
    keynameoffset: int,
    allow_recovery: bool,
) -> Dict[str, torch.Tensor]:
    """Load a ResNet20-style checkpoint for auxiliary variants.

    When ``allow_recovery`` is True and the checkpoint is missing, we reuse the
    generic ResNet20 recovery pipeline (torch.hub → auto-train) for CIFAR
    datasets. Manipulated checkpoints set ``allow_recovery`` to False so that we
    surface a clear FileNotFoundError instead of silently training.
    """

    try:
        checkpoint: Any = torch.load(path, map_location=device)
    except (FileNotFoundError, OSError):
        if allow_recovery and dataset_enum in (DatasetEnum.CIFAR10, DatasetEnum.CIFAR100):
            checkpoint = _recover_resnet20_checkpoint(path, device, num_classes, dataset_enum.name.lower())
        else:
            raise

    if checkpoint is None:
        raise FileNotFoundError(f"Unable to recover checkpoint for path '{path}'.")

    state: Any
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint

    if isinstance(state, dict) and keynameoffset and any(key.startswith('module.') for key in state):
        state = {key[keynameoffset:]: value for key, value in state.items()}

    # Ensure tensors live on CPU before loading into target model; the caller will move the model afterwards.
    if isinstance(state, dict):
        state = {key: tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor for key, tensor in state.items()}

    return state  # type: ignore[return-value]


def _transfer_matching_weights(model: nn.Module, source_state: Dict[str, torch.Tensor], device: torch.device) -> int:
    target_state = model.state_dict()
    transferred = 0
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value.to(target_state[key].device if hasattr(target_state[key], 'device') else device)
            transferred += 1
    model.load_state_dict(target_state, strict=False)
    return transferred


def _maybe_run_warmup(
    model: torch.nn.Module,
    dataset_enum: DatasetEnum,
    device: torch.device,
    num_classes: int,
    dataset_key: str,
    arch_key: str,
) -> torch.nn.Module:
    """Optionally fine-tune freshly loaded models for a few warm-up epochs."""

    if not is_warmup_enabled():
        return model.eval().to(device)

    epochs, lr, batch_size = get_warmup_config()
    if epochs <= 0 or lr <= 0 or batch_size <= 0:
        return model.eval().to(device)

    verbose = os.getenv('VERBOSE_MODEL_LOAD', '1') == '1'

    try:
        from load import load_data_loaders
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[warm-up] skipped for {dataset_key}:{arch_key} (missing data loader imports: {exc})")
        return model.eval().to(device)

    try:
        train_loader, _ = load_data_loaders(
            dataset_enum,
            train_batch_size=batch_size,
            test_batch_size=batch_size,
            train_limit=None,
            test_limit=0,
            test_only=False,
            shuffle_train=True,
            shuffle_test=False,
        )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[warm-up] unable to build loaders for {dataset_key}:{arch_key} -> {exc}")
        return model.eval().to(device)

    if train_loader is None:
        if verbose:
            print(f"[warm-up] no training loader available for {dataset_key}:{arch_key}")
        return model.eval().to(device)

    model = model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        if verbose:
            mean_loss = total_loss / max(steps, 1)
            print(f"[warm-up {dataset_key}:{arch_key}] epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}")

    model.eval()
    return model.to(device)


def _load_vgg_generic(
    path: str,
    device: torch.device,
    *,
    dataset_enum: DatasetEnum,
    num_classes: int,
    builder: Callable[[int], nn.Module],
    hub_name: str,
    save_prefix: str,
    train_prefix: str,
) -> nn.Module:
    import os
    import torch
    import torch.nn as nn  # noqa: F401  (kept for potential future extensions)
    from torch.utils.data import DataLoader, TensorDataset
    from load import load_data

    verbose = os.getenv('VERBOSE_MODEL_LOAD', '1') == '1'

    model = builder(num_classes).to(device)

    try:
        raw_checkpoint = torch.load(path, map_location=device)
        state_dict = raw_checkpoint['state_dict'] if isinstance(raw_checkpoint, dict) and 'state_dict' in raw_checkpoint else raw_checkpoint
        model.load_state_dict(state_dict, strict=True)
        return model.eval()
    except (FileNotFoundError, OSError, KeyError):
        pass

    tv_state = None
    if os.getenv('TRY_TORCH_HUB_VGG', '1') == '1':
        try:
            tv_model = torch.hub.load('pytorch/vision', hub_name, pretrained=True)
            tv_state = tv_model.state_dict()
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[load_{save_prefix}] torch.hub fetch failed: {exc}")

    auto_train_allowed = is_auto_train_enabled()

    transfer_success = False
    if tv_state is not None:
        try:
            transferred = _transfer_matching_weights(model, tv_state, device)
            transfer_success = transferred > 0
            if verbose:
                print(f"[load_{save_prefix}] transferred {transferred} tensors from torchvision hub model")
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[load_{save_prefix}] hub weight transfer failed: {exc}")
            transfer_success = False
            tv_state = None

    if not auto_train_allowed:
        if not transfer_success:
            raise FileNotFoundError(
                f"Missing checkpoint at '{path}' for dataset {dataset_enum.name} and auto-training disabled via config."
            )
        return model.eval()

    epochs, lr, batch_size = get_train_config(30, 1e-3, 256, specific_prefix=train_prefix)
    x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1
        if verbose:
            mean_loss = epoch_loss / max(steps, 1)
            print(f"[load_{save_prefix} auto-train] epoch {ep + 1}/{epochs} loss={mean_loss:.4f}")

    @torch.no_grad()
    def _evaluate(acc_model: torch.nn.Module) -> float:
        acc_model.eval()
        correct = 0
        total = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = acc_model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    accuracy = _evaluate(model)

    ckpt_dir = os.path.dirname(path)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'accuracy.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
    torch.save({
        'state_dict': model.state_dict(),
        'meta': {
            'source': f'{save_prefix}_local',
            'dataset': dataset_enum.name.lower(),
            'num_classes': num_classes,
            'accuracy': accuracy,
        },
    }, path)
    if verbose:
        print(f"[load_{save_prefix} auto-train] saved checkpoint to {path} acc={accuracy:.4f}")

    return model.eval()


# --------------------------------------------------------------------------------------
# Public helpers built on top of the ArtifactManager
# --------------------------------------------------------------------------------------

def load_models(which: str, n=10):
    """
    Loads n trained models of type which.

    :rtype: list
    :raises Exception:  model number i not found
    """
    modellist = []
    for i in tqdm.tqdm(range(n)):
        modellist.append(load_model(which, i))
    return modellist


def load_model(which: str, i: int):
    """Thin wrapper: parse <dataset>_<arch> (supports legacy aliases) and delegate to ArtifactManager."""
    device = torch.device(os.getenv('CUDADEVICE'))
    dataset_key, arch_key = _normalize_model_string(which)
    manager = _get_artifact_manager()
    return manager.load_clean(dataset_key, arch_key, i, device)


def load_manipulated_model(model_root, which: str):
    """Thin wrapper: parse <dataset>_<arch> (supports legacy aliases) and delegate to ArtifactManager for attacked models."""
    print("model root", model_root)
    device = torch.device(os.getenv('CUDADEVICE'))
    dataset_key, arch_key = _normalize_model_string(which)
    manager = _get_artifact_manager()
    return manager.load_manipulated(model_root, dataset_key, arch_key, device)

def _log_resnet20_recovery(message: str) -> None:
    if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
        print(f"[resnet20-auto] {message}")


def _load_resnet20_from_torchhub(device, num_classes: int) -> Optional[Dict[str, Any]]:
    if num_classes != 10:
        return None
    try:
        hub_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
    except Exception as exc:  # noqa: BLE001
        _log_resnet20_recovery(f"torch.hub load failed: {exc}")
        return None

    try:
        import importlib
        resnet_module = importlib.import_module('models.resnet')
        model = resnet_module.resnet20(num_classes=num_classes)
    except Exception as exc:  # noqa: BLE001
        _log_resnet20_recovery(f"local ResNet20 build failed: {exc}")
        return None

    hub_state = hub_model.state_dict()
    missing, unexpected = model.load_state_dict(hub_state, strict=False)
    if missing or unexpected:
        _log_resnet20_recovery(f"state_dict mismatch (missing={missing}, unexpected={unexpected})")
        return None

    model = model.to(device).eval()
    accuracy = None
    loss = None
    try:
        from train.model_trainer import evaluate_model, get_cifar10_loaders
        _, test_loader = get_cifar10_loaders()
        if test_loader is not None:
            accuracy, loss = evaluate_model(model, test_loader, device)
    except Exception as exc:  # noqa: BLE001
        _log_resnet20_recovery(f"evaluation of hub weights failed: {exc}")

    state_dict_cpu = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    return {
        'state_dict': state_dict_cpu,
        'meta': {
            'source': 'torchhub:chenyaofo/pytorch-cifar-models',
            'dataset': 'cifar10',
            'num_classes': num_classes,
            'accuracy': accuracy,
            'val_loss': loss,
        },
    }


def _train_resnet20_and_save(path: str, device, num_classes: int, dataset: DatasetEnum) -> Dict[str, Any]:
    """Auto-train a ResNet20 checkpoint for the requested dataset and persist it."""
    if not is_auto_train_enabled():
        raise RuntimeError("Auto-training disabled via config (train_on_requested_dataset=false).")
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    import torch.optim as optim
    import torch.nn.functional as F
    import torch
    from load import load_data

    if dataset not in (DatasetEnum.CIFAR10, DatasetEnum.CIFAR100):
        raise RuntimeError(f'Automatic ResNet20 training not supported for dataset {dataset}.')

    # Hyperparameters via environment overrides
    epochs, lr, batch_size = get_train_config(60, 1e-3, 256, specific_prefix='RESNET20_AUTO')

    # Ensure output directory exists
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    # Load tensors (already normalized by dataset loader)
    x_test, y_test, x_train, y_train = load_data(dataset, test_only=False, shuffle_test=True)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

    model = resnet_normal.resnet20(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
            mean_loss = total_loss / max(steps, 1)
            print(f"[resnet20 auto-train {dataset.name.lower()}] epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}")

    @torch.no_grad()
    def _evaluate(acc_model: torch.nn.Module) -> float:
        acc_model.eval()
        correct = 0
        total = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = acc_model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    accuracy = _evaluate(model)
    saved_state = {
        'state_dict': model.state_dict(),
        'meta': {
            'source': f'resnet20_local_{dataset.name.lower()}',
            'dataset': dataset.name.lower(),
            'num_classes': num_classes,
            'accuracy': accuracy,
        },
    }
    torch.save(saved_state, path)
    with open(os.path.join(folder, 'accuracy.txt'), 'w') as f:
        f.write(f"{accuracy:.4f}\n")

    _log_resnet20_recovery(f"trained new {dataset.name} checkpoint saved to {path} (acc={accuracy:.4f})")
    return saved_state


def _recover_resnet20_checkpoint(path: str, device, num_classes: int, dataset_hint: str) -> Optional[Dict[str, Any]]:
    dataset_hint = (dataset_hint or 'unknown').lower()
    if dataset_hint not in {'cifar10', 'cifar100'}:
        raise FileNotFoundError(f"Missing checkpoint at '{path}' and automatic recovery unavailable for dataset '{dataset_hint}'.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset_enum = DatasetEnum.CIFAR10 if dataset_hint == 'cifar10' else DatasetEnum.CIFAR100
    checkpoint = None
    if dataset_enum == DatasetEnum.CIFAR10:
        checkpoint = _load_resnet20_from_torchhub(device, num_classes)
        if checkpoint is not None:
            torch.save(checkpoint, path)
            _log_resnet20_recovery(f"downloaded ResNet20 weights from torch.hub into {path}")
            return checkpoint
    _log_resnet20_recovery('torch.hub weights unavailable, falling back to training from scratch')
    if not is_auto_train_enabled():
        raise FileNotFoundError(
            f"Missing checkpoint at '{path}' and auto-training disabled via config (train_on_requested_dataset=false)."
        )
    return _train_resnet20_and_save(path, device, num_classes, dataset_enum)


# --------------------------------------------------------------------------------------
# Spec registry and alias normalization
# --------------------------------------------------------------------------------------

def _dataset_specs() -> Dict[str, DatasetSpec]:
    return {k: DatasetSpec(k, enum, n) for k, (enum, n) in _DATASET_METADATA.items()}


def _arch_specs() -> Dict[str, ModelSpec]:
    """Register ModelSpec per architecture. Path rules encoded per-arch for consistency with repo layout."""

    def resnet20_path(dataset_key: str, index: int) -> Tuple[str, str]:
        if dataset_key == 'cifar10':
            return (os.path.join('models', 'cifar10_resnet20'), f'model_{index}.th')
        if dataset_key == 'cifar100':
            return (os.path.join('models', 'cifar100_resnet20_normal'), f'model_{index}.th')
        if dataset_key == 'gtsrb':
            return (os.path.join('models', 'gtsrb_resnet'), f'model_{index}.th')
        raise Exception(f"Unsupported dataset '{dataset_key}' for resnet20")

    def vgg13_path(dataset_key: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', f'{dataset_key}_vgg13'), f'model_{index}.th')

    def vgg13bn_path(dataset_key: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', f'{dataset_key}_vgg13bn'), f'model_{index}.th')

    def wrn_path(dataset_key: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', f'{dataset_key}_wideresnet28_10'), f'model_{index}.th')

    def mnv3_path(dataset_key: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', f'{dataset_key}_mobilenetv3small'), f'model_{index}.th')

    def vit_path(dataset_key: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', f'{dataset_key}_vit_b_16'), f'model_{index}.th')

    def vit_bn_path(dataset_key: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', f'{dataset_key}_vit_b_16bn'), f'model_{index}.th')

    def imgnet_r18_path(_: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', 'imagenet_resnet18_normal'), f'model_{index}.th')

    def imgnet_r18_xbn_path(_: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', 'imagenet_resnet18_xbn'), f'model_{index}.th')

    def imgnet_r50_path(_: str, index: int) -> Tuple[str, str]:
        return (os.path.join('models', 'imagenet_resnet50_normal'), f'model_{index}.pth')

    def _load_resnet20_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        if dataset == DatasetEnum.GTSRB:
            return load_gtsrb_model_normal(path, device, state_dict=True, keynameoffset=7, num_classes=num_classes)
        return load_resnet20_model_normal(path, device, state_dict=True, keynameoffset=7, num_classes=num_classes)

    def _load_resnet20_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        if dataset == DatasetEnum.GTSRB:
            return load_gtsrb_model_normal(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)
        return load_resnet20_model_normal(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)

    def _load_resnet20_bn_drop_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_resnet20_model_bn_drop_org(path, device, state_dict=True, keynameoffset=7, num_classes=num_classes)

    def _load_resnet20_bn_drop_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_resnet20_model_bn_drop(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)

    def _load_resnet20_cfn_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_resnet20_model_cfn(path, device, state_dict=True, keynameoffset=7, num_classes=num_classes)

    def _load_resnet20_cfn_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_resnet20_model_cfn(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)

    def _load_resnet_freeze_bn_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_resnet20_model_freeze_bn(path, device, state_dict=True, keynameoffset=7, num_classes=num_classes)

    def _load_resnet_freeze_bn_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_resnet20_model_freeze_bn(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)

    def _load_resnet_nbn_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_resnet20_model_nbn(path, device, state_dict=True, keynameoffset=7, num_classes=num_classes)

    def _load_resnet_nbn_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_resnet20_model_nbn(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)

    def _load_vgg13_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_vgg13(path, device, num_classes=num_classes)

    def _load_vgg13_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_vgg13_attacked(path, device, keynameoffset=7, num_classes=num_classes)

    def _load_vgg13bn_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_vgg13bn(path, device, num_classes=num_classes)

    def _load_vgg13bn_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_vgg13bn_attacked(path, device, keynameoffset=7, num_classes=num_classes)

    def _load_wrn_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        wrn_depth = int(os.getenv('WRN_DEPTH', '28'))
        wrn_widen = int(os.getenv('WRN_WIDEN_FACTOR', '10'))
        wrn_drop = float(os.getenv('WRN_DROPRATE', '0.0'))
        return load_wideresnet_model_normal(path, device, num_classes=num_classes, dropRate=wrn_drop, depth=wrn_depth, widen_factor=wrn_widen, dataset_enum=dataset)

    def _load_wrn_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return _load_wrn_clean(path, device, dataset, num_classes)

    def _load_mnv3_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_mobilenetv3small_model_normal(path, device, num_classes=num_classes, dataset_enum=dataset)

    def _load_mnv3_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return _load_mnv3_clean(path, device, dataset, num_classes)

    def _load_vit_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_vit_b_16_model_normal(path, device, num_classes=num_classes, dataset_enum=dataset)

    def _load_vit_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return _load_vit_clean(path, device, dataset, num_classes)

    def _load_vit_bn_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_vit_b_16bn_model_normal(path, device, num_classes=num_classes, dataset_enum=dataset)

    def _load_vit_bn_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_vit_b_16bn_model_manipulated(path, device, num_classes=num_classes, dataset_enum=dataset)

    def _load_imagenet_r18_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_imagenet_resnet18_model_local(path=path, device=device)

    def _load_imagenet_r18_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_imagenet_resnet18_manipulated(path, device=device)

    def _load_imagenet_r18_xbn_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_imagenet_resnet18xbn_model_local(path=path, device=device)

    def _load_imagenet_r18_xbn_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_imagenet_resnet18xbn_manipulated(path, device=device)

    def _load_imagenet_r50_clean(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_imagenet_resnet50_model_local(path=path, device=device)

    def _load_imagenet_r50_manip(path: str, device: torch.device, dataset: DatasetEnum, num_classes: int):
        return load_imagenet_resnet50_manipulated(path, device=device)

    def _hub_resnet20(device: torch.device, num_classes: int, dataset: DatasetEnum, dataset_key: str):
        if dataset == DatasetEnum.CIFAR10:
            return _load_resnet20_from_torchhub(device, num_classes)
        return None

    specs: Dict[str, ModelSpec] = {
        'resnet20': ModelSpec(
            arch_key='resnet20',
            path_fn=resnet20_path,
            load_clean=_load_resnet20_clean,
            load_manipulated=_load_resnet20_manip,
            hub_fetcher=_hub_resnet20,
            train_fallback=lambda path, device, ncls, dset: _train_resnet20_and_save(path, device, ncls, dset),
        ),
        'resnet20_bn_drop': ModelSpec(
            arch_key='resnet20_bn_drop',
            path_fn=lambda dk, i: (os.path.join('models', 'cifar10_resnet20' if dk == 'cifar10' else f'{dk}_resnet20_normal'), f'model_{i}.th'),
            load_clean=_load_resnet20_bn_drop_clean,
            load_manipulated=_load_resnet20_bn_drop_manip,
        ),
        'resnet20_cfn': ModelSpec(
            arch_key='resnet20_cfn',
            path_fn=lambda dk, i: (os.path.join('models', 'cifar10_resnet20' if dk == 'cifar10' else f'{dk}_resnet20_normal'), f'model_{i}.th'),
            load_clean=_load_resnet20_cfn_clean,
            load_manipulated=_load_resnet20_cfn_manip,
        ),
        'resnet_freeze_bn': ModelSpec(
            arch_key='resnet_freeze_bn',
            path_fn=lambda dk, i: (os.path.join('models', 'cifar10_resnet_freeze_bn'), f'model_{i}.th'),
            load_clean=_load_resnet_freeze_bn_clean,
            load_manipulated=_load_resnet_freeze_bn_manip,
        ),
        'resnet_nbn': ModelSpec(
            arch_key='resnet_nbn',
            path_fn=lambda dk, i: (os.path.join('models', 'cifar10_resnet_nbn'), f'model_{i}.th'),
            load_clean=_load_resnet_nbn_clean,
            load_manipulated=_load_resnet_nbn_manip,
        ),
        'vgg13': ModelSpec(
            arch_key='vgg13',
            path_fn=vgg13_path,
            load_clean=_load_vgg13_clean,
            load_manipulated=_load_vgg13_manip,
        ),
        'vgg13bn': ModelSpec(
            arch_key='vgg13bn',
            path_fn=vgg13bn_path,
            load_clean=_load_vgg13bn_clean,
            load_manipulated=_load_vgg13bn_manip,
        ),
        'wideresnet28_10': ModelSpec(
            arch_key='wideresnet28_10',
            path_fn=wrn_path,
            load_clean=_load_wrn_clean,
            load_manipulated=_load_wrn_manip,
        ),
        'mobilenetv3small': ModelSpec(
            arch_key='mobilenetv3small',
            path_fn=mnv3_path,
            load_clean=_load_mnv3_clean,
            load_manipulated=_load_mnv3_manip,
        ),
        'vit_b_16': ModelSpec(
            arch_key='vit_b_16',
            path_fn=vit_path,
            load_clean=_load_vit_clean,
            load_manipulated=_load_vit_manip,
        ),
        'vit_b_16bn': ModelSpec(
            arch_key='vit_b_16bn',
            path_fn=vit_bn_path,
            load_clean=_load_vit_bn_clean,
            load_manipulated=_load_vit_bn_manip,
        ),
        'resnet18': ModelSpec(
            arch_key='resnet18',
            path_fn=imgnet_r18_path,
            load_clean=_load_imagenet_r18_clean,
            load_manipulated=_load_imagenet_r18_manip,
        ),
        'resnet18_xbn': ModelSpec(
            arch_key='resnet18_xbn',
            path_fn=imgnet_r18_xbn_path,
            load_clean=_load_imagenet_r18_xbn_clean,
            load_manipulated=_load_imagenet_r18_xbn_manip,
        ),
        'resnet50': ModelSpec(
            arch_key='resnet50',
            path_fn=imgnet_r50_path,
            load_clean=_load_imagenet_r50_clean,
            load_manipulated=_load_imagenet_r50_manip,
        ),
    }
    return specs


def _alias_map() -> Dict[str, str]:
    """Legacy aliases mapping to canonical <dataset>_<arch> pairs.

    Only map the architecture portion here; dataset is inferred from explicit prefix when present or defaults.
    """
    return {
        # resnet20 variants
        'resnet20_normal': 'resnet20',  # arch-only alias
        'resnet20': 'resnet20',  # idempotent
        'resnet20_gtsrb': 'resnet20',
        'resnet20_bn_drop': 'resnet20_bn_drop',
        'resnet20_cfn': 'resnet20_cfn',
        'resnet20_freeze_bn': 'resnet_freeze_bn',
        'resnet20_nbn': 'resnet_nbn',
        # vgg
        'vgg13_normal': 'vgg13',           # arch-only alias
        'vgg13bn_normal': 'vgg13bn',       # arch-only alias
        'vgg13': 'vgg13',
        'vgg13bn': 'vgg13bn',
        # imagenet
        'resnet18_normal': 'resnet18',     # arch-only alias
        'resnet50_normal': 'resnet50',     # arch-only alias
        'imagenet_resnet18_normal': 'resnet18',
        'imagenet_resnet18_xbn': 'resnet18_xbn',
        'imagenet_resnet50_normal': 'resnet50',
    }


_ARCH_SPECS_CACHE: Optional[Dict[str, ModelSpec]] = None
_DATASET_SPECS_CACHE: Optional[Dict[str, DatasetSpec]] = None


def _get_arch_specs() -> Dict[str, ModelSpec]:
    global _ARCH_SPECS_CACHE
    if _ARCH_SPECS_CACHE is None:
        _ARCH_SPECS_CACHE = _arch_specs()
    return _ARCH_SPECS_CACHE


def _get_dataset_specs() -> Dict[str, DatasetSpec]:
    global _DATASET_SPECS_CACHE
    if _DATASET_SPECS_CACHE is None:
        _DATASET_SPECS_CACHE = _dataset_specs()
    return _DATASET_SPECS_CACHE


_ARTIFACT_MANAGER_CACHE: Optional[ArtifactManager] = None


def _get_artifact_manager() -> ArtifactManager:
    global _ARTIFACT_MANAGER_CACHE
    if _ARTIFACT_MANAGER_CACHE is None:
        _ARTIFACT_MANAGER_CACHE = ArtifactManager(_get_dataset_specs(), _get_arch_specs())
    return _ARTIFACT_MANAGER_CACHE


def _normalize_model_string(which: str) -> Tuple[str, str]:
    """Return (dataset_key, arch_key) from possibly aliased string.

    Accepted forms:
      - canonical: "<dataset>_<arch>" e.g. "cifar10_resnet20"
      - legacy aliases like "resnet20_normal", optionally prefixed like "cifar100_vgg13bn_normal"
    """
    which_l = which.lower()
    parts = which_l.split('_')
    dspecs = _get_dataset_specs()
    alias = _alias_map()

    # Detect explicit dataset prefix
    if parts and parts[0] in dspecs:
        dataset_key = parts[0]
        arch_part = '_'.join(parts[1:]) if len(parts) > 1 else ''
        if arch_part in _get_arch_specs():
            return dataset_key, arch_part
        # Alias mapping for arch
        mapped = alias.get(arch_part)
        if mapped is not None:
            return dataset_key, mapped
        # Some legacy forms like imagenet_resnet50_normal already have dataset prefix
        if arch_part.startswith('resnet') or arch_part.startswith('vgg') or arch_part.startswith('vit') or arch_part.startswith('wideresnet') or arch_part.startswith('mobilenet'):
            # try to trim trailing "_normal" if present
            if arch_part.endswith('_normal'):
                arch_trim = arch_part[:-7]
                if arch_trim in _get_arch_specs():
                    return dataset_key, arch_trim
        # Fallback: treat remainder as arch key
        return dataset_key, arch_part

    # No explicit dataset: try to infer dataset from keywords in string
    inferred_dataset = None
    for ds in dspecs.keys():
        if ds in which_l:
            inferred_dataset = ds
            break
    if inferred_dataset is None:
        # default to CIFAR10 for ambiguous legacy aliases
        inferred_dataset = 'cifar10'

    # Map alias arch if necessary
    arch_key = which_l
    # If the string is like 'resnet20_normal' etc
    mapped = alias.get(which_l)
    if mapped is not None:
        arch_key = mapped
    else:
        # Drop dataset tokens if embedded
        for ds in dspecs.keys():
            if arch_key.startswith(ds + '_'):
                arch_key = arch_key[len(ds) + 1:]
                break
        # Trim trailing _normal for imagenet style
        if arch_key.endswith('_normal'):
            arch_key = arch_key[:-7]

    return inferred_dataset, arch_key


def load_resnet20_model_normal(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):
    assert(option == 'A' or option == 'B')
    model = resnet_normal.resnet20(**kwargs)

    path_lower = path.lower()
    if 'cifar100' in path_lower:
        dataset_hint = 'cifar100'
    elif 'cifar10' in path_lower:
        dataset_hint = 'cifar10'
    elif 'gtsrb' in path_lower:
        dataset_hint = 'gtsrb'
    else:
        dataset_hint = 'unknown'

    try:
        checkpoint = torch.load(path, map_location=device)
    except (FileNotFoundError, OSError):
        checkpoint = _recover_resnet20_checkpoint(path, device, kwargs.get('num_classes', 10), dataset_hint)

    if checkpoint is None:
        raise FileNotFoundError(f"Unable to recover checkpoint for path '{path}'.")

    base_state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    if isinstance(base_state, dict) and keynameoffset and any(key.startswith('module.') for key in base_state.keys()):
        trimmed_state = {key[keynameoffset:]: val for key, val in base_state.items()}
    else:
        trimmed_state = base_state

    if state_dict:
        model.load_state_dict(trimmed_state)
    else:
        model.load_state_dict(trimmed_state)

    return model.eval().to(device)


def load_resnet20_model_nbn(path, device, state_dict=False, option='A', keynameoffset=7, **kwargs):
    assert option in ('A', 'B')
    extra_kwargs = dict(kwargs)
    dataset_enum = extra_kwargs.pop('dataset_enum', None) or _infer_dataset_from_path(path)
    num_classes = extra_kwargs.get('num_classes', _default_num_classes(dataset_enum))
    extra_kwargs.setdefault('num_classes', num_classes)

    # resnet_nbn removes all batch norm layers; auto-recovered checkpoints from the
    # standard ResNet20 would not match this architecture, so we require an existing file.
    state = _load_resnet20_variant_state(
        path,
        device,
        dataset_enum=dataset_enum,
        num_classes=num_classes,
        keynameoffset=keynameoffset,
        allow_recovery=False,
    )

    model = resnet_nbn.resnet20(**extra_kwargs)
    model.load_state_dict(state, strict=state_dict)
    return model.eval().to(device)


def load_resnet20_model_bn_drop_org(path, device, state_dict=False, option='A', keynameoffset=7, **kwargs):
    assert option in ('A', 'B')
    extra_kwargs = dict(kwargs)
    dataset_enum = extra_kwargs.pop('dataset_enum', None) or _infer_dataset_from_path(path)
    num_classes = extra_kwargs.get('num_classes', _default_num_classes(dataset_enum))
    extra_kwargs.setdefault('num_classes', num_classes)

    state = _load_resnet20_variant_state(
        path,
        device,
        dataset_enum=dataset_enum,
        num_classes=num_classes,
        keynameoffset=keynameoffset,
        allow_recovery=state_dict,
    )

    model = resnet_normal.resnet20(**extra_kwargs)
    model.load_state_dict(state, strict=state_dict)
    remove_batchnorm(model)
    return model.eval().to(device)


def load_resnet20_model_bn_drop(path, device, state_dict=False, option='A', keynameoffset=7, **kwargs):
    assert option in ('A', 'B')
    extra_kwargs = dict(kwargs)
    dataset_enum = extra_kwargs.pop('dataset_enum', None) or _infer_dataset_from_path(path)
    num_classes = extra_kwargs.get('num_classes', _default_num_classes(dataset_enum))
    extra_kwargs.setdefault('num_classes', num_classes)

    state = _load_resnet20_variant_state(
        path,
        device,
        dataset_enum=dataset_enum,
        num_classes=num_classes,
        keynameoffset=keynameoffset,
        allow_recovery=False,
    )

    model = resnet_normal.resnet20(**extra_kwargs)
    model.load_state_dict(state, strict=False)
    remove_batchnorm(model)
    return model.eval().to(device)


def load_resnet20_model_cfn(path, device, state_dict=False, option='A', keynameoffset=7, **kwargs):
    assert option in ('A', 'B')
    extra_kwargs = dict(kwargs)
    dataset_enum = extra_kwargs.pop('dataset_enum', None) or _infer_dataset_from_path(path)
    num_classes = extra_kwargs.get('num_classes', _default_num_classes(dataset_enum))
    extra_kwargs.setdefault('num_classes', num_classes)

    state = _load_resnet20_variant_state(
        path,
        device,
        dataset_enum=dataset_enum,
        num_classes=num_classes,
        keynameoffset=keynameoffset,
        allow_recovery=state_dict,
    )

    model = resnet_normal.resnet20(**extra_kwargs)
    model.load_state_dict(state, strict=state_dict)
    model = replace_bn(model)
    return model.eval().to(device)


def load_resnet20_model_freeze_bn(path, device, state_dict=False, option='A', keynameoffset=7, **kwargs):
    assert option in ('A', 'B')
    extra_kwargs = dict(kwargs)
    dataset_enum = extra_kwargs.pop('dataset_enum', None) or _infer_dataset_from_path(path)
    num_classes = extra_kwargs.get('num_classes', _default_num_classes(dataset_enum))
    extra_kwargs.setdefault('num_classes', num_classes)

    state = _load_resnet20_variant_state(
        path,
        device,
        dataset_enum=dataset_enum,
        num_classes=num_classes,
        keynameoffset=keynameoffset,
        allow_recovery=state_dict,
    )

    model = resnet_freeze_bn.resnet20(**extra_kwargs)
    model.load_state_dict(state, strict=state_dict)
    return model.eval().to(device)


def load_gtsrb_model_normal(path, device, state_dict=False, option='A', keynameoffset=7, **kwargs):
    assert option in ('A', 'B')
    extra_kwargs = dict(kwargs)
    if 'num_classes' not in extra_kwargs:
        extra_kwargs['num_classes'] = 43

    try:
        checkpoint: Any = torch.load(path, map_location=device)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"Missing GTSRB checkpoint at '{path}'.") from exc

    state: Any
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint

    if isinstance(state, dict) and keynameoffset and any(key.startswith('module.') for key in state):
        state = {key[keynameoffset:]: value for key, value in state.items()}

    model = resnet_normal.resnet20(**extra_kwargs)
    model.load_state_dict(state, strict=state_dict)
    return model.eval().to(device)

def load_vgg13(path, device, state_dict=False, keynameoffset=7, **kwargs):
    """
    Load VGG13 from a local checkpoint. If the checkpoint is missing, auto-train
    a small model on the inferred dataset (CIFAR10 or GTSRB) and save it to `path`.

    Training can be controlled via env vars:
      VGG_AUTO_EPOCHS (default 30)
      VGG_AUTO_LR     (default 1e-3)
      VGG_AUTO_BS     (default 256)
    """
    import torch
    import torch.nn as nn
    import os
    from torch.utils.data import DataLoader, TensorDataset
    from load import load_data
    # Try to load checkpoint first
    try:
        d = torch.load(path, map_location=device)['state_dict']
        model = vgg.vgg13(**kwargs)
        model.load_state_dict(d)
        return model.eval().to(device)
    except (FileNotFoundError, OSError, KeyError):
        # Inference of dataset from path; fallback to CIFAR10
        lower_path = path.lower()
        if 'gtsrb_' in lower_path or 'gtsrb/' in lower_path:
            dataset_enum = DatasetEnum.GTSRB
        elif 'cifar100' in lower_path:
            dataset_enum = DatasetEnum.CIFAR100
        else:
            dataset_enum = DatasetEnum.CIFAR10

        if not is_auto_train_enabled():
            raise FileNotFoundError(
                f"Missing checkpoint at '{path}' for dataset {dataset_enum.name} and auto-training disabled via config."
            )

        if 'num_classes' in kwargs:
            num_classes = kwargs['num_classes']
        elif dataset_enum == DatasetEnum.CIFAR100:
            num_classes = 100
        elif dataset_enum == DatasetEnum.GTSRB:
            num_classes = 43
        else:
            num_classes = 10

        epochs, lr, batch_size = get_train_config(30, 1e-3, 256, specific_prefix='VGG_AUTO')

        # Load tensors
        x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

        # Build model and train
        model = vgg.vgg13(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            loss_sum = 0.0
            steps = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
            if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
                print(f"[vgg13 auto-train] epoch {ep+1}/{epochs} loss={(loss_sum/max(steps,1)):.4f}")

        @torch.no_grad()
        def _acc(m):
            m.eval()
            correct = 0
            total = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            return correct / max(total, 1)

        acc = _acc(model)
        # Ensure directory exists and save
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, 'accuracy.txt'), 'w') as f:
            f.write(f"Accuracy: {acc:.4f}\n")
        torch.save({'state_dict': model.state_dict(), 'meta': {'source': 'vgg13_local', 'num_classes': num_classes, 'accuracy': acc}}, path)
        if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
            print(f"[vgg13 auto-train] saved checkpoint to {path} acc={acc:.4f}")
        return model.eval().to(device)
def load_vgg13_attacked(path, device, state_dict=False,keynameoffset=7,**kwargs):
    #assert(option == 'A' or option == 'B')
    d = torch.load(path, map_location=device)
    model = vgg.vgg13(**kwargs)
    model.load_state_dict(d)
    return model.eval().to(device)

def load_vgg13bn(path, device, state_dict=False, keynameoffset=7, **kwargs):
    """
    Load VGG13-BN from checkpoint at `path`. If missing, auto-train and save.

    Env vars:
      VGG_AUTO_EPOCHS (default 30)
      VGG_AUTO_LR     (default 1e-3)
      VGG_AUTO_BS     (default 256)
    """
    import torch
    import torch.nn as nn
    import os
    from torch.utils.data import DataLoader, TensorDataset
    from load import load_data
    try:
        d = torch.load(path, map_location=device)['state_dict']
        model = vgg.vgg13_bn(**kwargs)
        model.load_state_dict(d, strict=True)
        return model.eval().to(device)
    except (FileNotFoundError, OSError, KeyError):
        lower_path = path.lower()
        if 'gtsrb_' in lower_path or 'gtsrb/' in lower_path:
            dataset_enum = DatasetEnum.GTSRB
        elif 'cifar100' in lower_path:
            dataset_enum = DatasetEnum.CIFAR100
        else:
            dataset_enum = DatasetEnum.CIFAR10

        if not is_auto_train_enabled():
            raise FileNotFoundError(
                f"Missing checkpoint at '{path}' for dataset {dataset_enum.name} and auto-training disabled via config."
            )

        if 'num_classes' in kwargs:
            num_classes = kwargs['num_classes']
        elif dataset_enum == DatasetEnum.CIFAR100:
            num_classes = 100
        elif dataset_enum == DatasetEnum.GTSRB:
            num_classes = 43
        else:
            num_classes = 10

        epochs, lr, batch_size = get_train_config(30, 1e-3, 256, specific_prefix='VGG_AUTO')

        x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

        model = vgg.vgg13_bn(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            loss_sum = 0.0
            steps = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
            if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
                print(f"[vgg13_bn auto-train] epoch {ep+1}/{epochs} loss={(loss_sum/max(steps,1)):.4f}")

        @torch.no_grad()
        def _acc(m):
            m.eval()
            correct = 0
            total = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            return correct / max(total, 1)

        acc = _acc(model)
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, 'accuracy.txt'), 'w') as f:
            f.write(f"Accuracy: {acc:.4f}\n")
        torch.save({'state_dict': model.state_dict(), 'meta': {'source': 'vgg13_bn_local', 'num_classes': num_classes, 'accuracy': acc}}, path)
        if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
            print(f"[vgg13_bn auto-train] saved checkpoint to {path} acc={acc:.4f}")
        return model.eval().to(device)
def load_vgg13bn_attacked(path, device, state_dict=False,keynameoffset=7,**kwargs):
    #assert(option == 'A' or option == 'B')
    d = torch.load(path, map_location=device)
    model = vgg.vgg13_bn(**kwargs)
    model.load_state_dict(d,strict=True)
    return model.eval().to(device)

def load_vgg13bn(path, device, state_dict=False, keynameoffset=7, **kwargs):
    """
    Load VGG13-BN with the standard fallback order (local → Torch Hub → auto-train).

    Shares the same environment controls as ``load_vgg13`` and respects the
    ``train_on_requested_dataset`` toggle before launching auto-training.
    """
    dataset_enum = _infer_dataset_from_path(path)
    extra_kwargs = dict(kwargs)
    num_classes = extra_kwargs.pop('num_classes', None)
    if num_classes is None:
        num_classes = _default_num_classes(dataset_enum)

    def _builder(nc: int) -> nn.Module:
        return vgg.vgg13_bn(num_classes=nc, **extra_kwargs)

    return _load_vgg_generic(
        path,
        device,
        dataset_enum=dataset_enum,
        num_classes=num_classes,
        builder=_builder,
        hub_name='vgg13_bn',
        save_prefix='vgg13_bn',
        train_prefix='VGG_AUTO',
    )
    model = resnet_normal.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        model.load_state_dict(d)
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d)

    return model.eval().to(device)
def load_wideresnet_model_normal(path, device, num_classes=10, dropRate=0.0, depth=28, widen_factor=10, dataset_enum: DatasetEnum = DatasetEnum.CIFAR10):
    """
    Load WideResNet weights from .th/.pth file.
    Automatically ignores final FC mismatch if needed.
    """
    from models.wideresnet import wideresnet28_10   # local factory supports depth & widen_factor
    # Fallback trainer if checkpoint is missing
    from models.auto_trainer import ensure_checkpoint_or_train
    # dataset_enum provided by caller; used for auto-train fallback

    def _try_torch_hub_wrn(depth_: int, widen_: int, num_classes_: int, device_: torch.device):
        """Optionally try to pull a WRN from torch.hub.

        Enabled when env TRY_TORCH_HUB_WRN == '1'. If a hub model is found,
        adapt its final classifier to num_classes and return it. Otherwise None.
        """
        if os.getenv("TRY_TORCH_HUB_WRN", "1") != "1":
            return None
        try:
            import torch as _torch
            # Known official torchvision WRNs are ImageNet-wide_resnet50_2/101_2, not CIFAR WRN-28-10.
            # We probe a couple of likely names; failures just fall through to local model/training.
            hub_specs = [
                ("pytorch/vision", f"wide_resnet{depth_}_{widen_}"),           # very likely missing
                # Add more third-party repos here if desired (kept empty by default)
            ]
            for repo, name in hub_specs:
                try:
                    m = _torch.hub.load(repo, name, pretrained=True)
                    # Try to adjust classifier head
                    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
                        in_f = m.fc.in_features
                        m.fc = nn.Linear(in_f, num_classes_)
                    elif hasattr(m, "linear") and isinstance(m.linear, nn.Linear):
                        in_f = m.linear.in_features
                        m.linear = nn.Linear(in_f, num_classes_)
                    return m.to(device_)
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # load checkpoint (or auto-create if missing)
    try:
        checkpoint = torch.load(path, map_location=device)
        ensured_path = path
    except (FileNotFoundError, OSError):
        # First optionally try hub
        hub_model = _try_torch_hub_wrn(depth, widen_factor, num_classes, device)
        if hub_model is not None:
            model = hub_model
            checkpoint = None
        else:
            if not is_auto_train_enabled():
                raise FileNotFoundError(
                    f"Missing checkpoint at '{path}' and auto-training disabled via config (train_on_requested_dataset=false)."
                )
            # Assume CIFAR10 for this loader's typical usage → train a quick local model
            epochs, lr, batch_size = get_train_config(5, 1e-3, 512, specific_prefix='WRN_AUTO')
            ensured_path = ensure_checkpoint_or_train(
                expected_file_path=path,
                dataset=dataset_enum,
                device=device,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                save_as="model_0.th",
                depth=depth,
                widen_factor=widen_factor,
            )
            checkpoint = torch.load(ensured_path, map_location=device)
    
    # Always assume local WiderResNet-28-10 checkpoint
    if 'model' not in locals():
        model = wideresnet28_10(num_classes=num_classes, dropRate=dropRate, depth=depth, widen_factor=widen_factor).to(device)

    # Some code paths may not have a checkpoint (hub case)
    if checkpoint is not None:
        # Some checkpoints store dict under "state_dict"
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # If FC layer size differs → remove last layer weights
        for key in ["linear.weight", "linear.bias"]:
            if key in checkpoint and checkpoint[key].shape != model.state_dict()[key].shape:
                print(f"[load_wideresnet] removing mismatched key: {key}")
                checkpoint.pop(key)

        # Load parameters
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model

def load_mobilenetv3small_model_normal(path, device, num_classes=10, dataset_enum: DatasetEnum = DatasetEnum.CIFAR10):
    """
    Load MobileNetV3-Small weights from disk with recovery fallbacks.

    Fallback order:
      1. Load checkpoint stored at ``path`` if present.
      2. Otherwise try Torch Hub (env ``TRY_TORCH_HUB_MNV3`` == ``'1'``). When successful,
         transfer torchvision weights into our ``mobilenet_v3_small`` wrapper and train on
         the requested dataset before persisting the checkpoint.
      3. If Torch Hub is disabled or unavailable, train the wrapper from scratch and save it.

    Training hyperparameters are retrieved via ``get_train_config`` with the
    ``MNV3_AUTO`` prefix (defaults: epochs=100, lr=1e-4, batch_size=512).
    """
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from load import load_data

    verbose = os.getenv("VERBOSE_MODEL_LOAD", "1") == "1"

    def _fetch_torchvision_model():
        if os.getenv("TRY_TORCH_HUB_MNV3", "1") != "1":
            return None
        try:
            tv_model = torch.hub.load('pytorch/vision', 'mobilenet_v3_small', pretrained=True)
            return tv_model
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[load_mobilenetv3small] torch.hub fetch failed: {exc}")
            return None

    def _load_checkpoint_into_model(model: torch.nn.Module, checkpoint_obj: Any):
        if isinstance(checkpoint_obj, dict) and 'state_dict' in checkpoint_obj:
            state_dict = checkpoint_obj['state_dict']
        else:
            state_dict = checkpoint_obj
        # Clone to avoid mutating the original mapping
        state_dict = {k: v for k, v in state_dict.items()}
        model_state = model.state_dict()
        for key in list(state_dict.keys()):
            if key in model_state and model_state[key].shape != state_dict[key].shape:
                if verbose:
                    print(f"[load_mobilenetv3small] removing mismatched key: {key}")
                state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)

    model = mobilenet_v3_small(num_classes=num_classes).to(device)

    try:
        checkpoint = torch.load(path, map_location='cpu')
    except (FileNotFoundError, OSError):
        checkpoint = None

    if checkpoint is not None:
        _load_checkpoint_into_model(model, checkpoint)
        model.eval()
        return model

    auto_train_allowed = is_auto_train_enabled()
    tv_model = _fetch_torchvision_model()

    if not auto_train_allowed:
        if tv_model is None:
            raise FileNotFoundError(
                f"Missing checkpoint at '{path}' and auto-training disabled via config (train_on_requested_dataset=false)."
            )
        try:
            tv_model = tv_model.to(device)
            transferred = transfer_from_torchvision_mnv3_small(model, tv_model)
            if verbose:
                print(f"[mobilenet_v3_small init] transferred {transferred} tensors from torchvision hub model")
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                "Torch Hub MobileNetV3-Small weights unavailable for initialization and auto-training is disabled."
            ) from exc
        model.eval()
        return model

    # No checkpoint found → build dataset loaders and train (with optional hub initialiser)
    epochs, lr, batch_size = get_train_config(100, 1e-4, 512, specific_prefix='MNV3_AUTO')
    x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

    if tv_model is not None:
        try:
            tv_model = tv_model.to(device)
            transferred = transfer_from_torchvision_mnv3_small(model, tv_model)
            if verbose:
                print(f"[mobilenet_v3_small init] transferred {transferred} tensors from torchvision hub model")
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[mobilenet_v3_small init] transfer failed: {exc}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        if verbose:
            mean_loss = total_loss / max(batches, 1)
            print(f"[mobilenet_v3_small auto-train] epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}")

    @torch.no_grad()
    def _evaluate(acc_model: torch.nn.Module) -> float:
        acc_model.eval()
        correct = 0
        total = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = acc_model(xb)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    accuracy = _evaluate(model)

    ckpt_dir = os.path.dirname(path)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'meta': {
            'source': 'mobilenet_v3_small_auto',
            'dataset': dataset_enum.name.lower(),
            'num_classes': num_classes,
            'accuracy': accuracy,
        },
    }, path)
    with open(os.path.join(ckpt_dir, 'accuracy.txt'), 'w') as f:
        f.write(f"{accuracy:.4f}\n")
    if verbose:
        print(f"[mobilenet_v3_small auto-train] saved checkpoint to {path} acc={accuracy:.4f}")

    model.eval()
    return model

def load_vit_b_16_model_normal(path, device, num_classes=10, dataset_enum: DatasetEnum = DatasetEnum.CIFAR10):
    """
    Load ViT-B/16 (CIFAR-10) wrapper weights from checkpoint or auto-train if missing.
    Fallback chain:
      1. Load local checkpoint at `path` if present.
      2. If missing and TRY_TORCH_HUB_VIT=1, attempt torch hub / torchvision.
         (Hub weights are not mapped; we still train local wrapper for consistency.)
      3. Quick auto-training of wrapper on CIFAR-10 then save checkpoint.

    Env overrides for wrapper & training:
      VIT_IMG_SIZE (default 32), VIT_PATCH_SIZE (4), VIT_EMBED_DIM (768), VIT_DEPTH (12),
      VIT_NUM_HEADS (12), VIT_MLP_RATIO (4.0), VIT_DROP (0.0),
      VIT_AUTO_EPOCHS (5), VIT_AUTO_LR (1e-3), VIT_AUTO_BS (256)
    """
    import torch
    import torch.nn as nn
    import os
    from load import load_data
    from torch.utils.data import TensorDataset, DataLoader

    # Use torchvision ViT-B/16 canonical defaults unless overridden
    # (224 image size, 16 patch size). These align with pretrained weights.
    img_size = int(os.getenv("VIT_IMG_SIZE", "32"))
    patch_size = int(os.getenv("VIT_PATCH_SIZE", "4"))
    embed_dim = int(os.getenv("VIT_EMBED_DIM", "768"))
    depth = int(os.getenv("VIT_DEPTH", "12"))
    num_heads = int(os.getenv("VIT_NUM_HEADS", "12"))
    mlp_ratio = float(os.getenv("VIT_MLP_RATIO", "4.0"))
    drop = float(os.getenv("VIT_DROP", "0.0"))
    epochs, lr, batch_size = get_train_config(30, 1e-4, 256, specific_prefix='VIT_AUTO')

    checkpoint = None
    try:
        checkpoint = torch.load(path, map_location=device)
    except (FileNotFoundError, OSError):
        # Optional hub attempt (not weight-mapped)
        if os.getenv("TRY_TORCH_HUB_VIT", "1") == "1":
            try:
                import torchvision
                _ = getattr(torchvision.models, 'vit_b_16', None)
            except Exception:
                pass
        if not is_auto_train_enabled():
            raise FileNotFoundError(
                f"Missing checkpoint at '{path}' for dataset {dataset_enum.name} and auto-training disabled via config."
            )
        # Auto-train (dataset-aware)
        x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
        # Build wrapper model for auto-training
        model = vit_b_16(num_classes=num_classes,
                         img_size=img_size,
                         patch_size=patch_size,
                         embed_dim=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         drop_rate=drop,
                         attn_drop_rate=0.0,
                         drop_path_rate=0.0).to(device)
        # ----- Optional: initialize from torchvision ViT-B_16 pretrained weights (ImageNet) -----
        def _try_load_torchvision_vit(device_: torch.device):
            try:
                import torchvision
                # Use weights enum if available (torchvision >=0.13)
                weights_attr = getattr(torchvision.models, 'ViT_B_16_Weights', None)
                if weights_attr is not None:
                    weights = weights_attr.DEFAULT
                    tv = torchvision.models.vit_b_16(weights=weights).to(device_)
                else:
                    # Fallback hub style
                    tv = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True).to(device_)
                tv.eval()
                return tv
            except Exception:
                return None

        tv_model = _try_load_torchvision_vit(device)
        if tv_model is not None:
            n = transfer_from_torchvision_vit(model, tv_model)
            if os.getenv("VERBOSE_MODEL_LOAD", "1") == "1":
                print(f"[vit_b_16 init] transferred {n} parameter tensors from torchvision pretrained model")
        criterion = nn.CrossEntropyLoss()
        # Allow training only the classification head via env flag
        head_only = os.getenv("VIT_TRAIN_HEAD_ONLY", "0") == "1"
        if head_only:
            # Freeze all but the final classifier
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
            head_lr = float(os.getenv("VIT_HEAD_LR", str(lr)))
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=head_lr)
            if os.getenv("VERBOSE_MODEL_LOAD", "0") == "1":
                print(f"[vit_b_16 auto-train] head-only training enabled (lr={head_lr})")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            loss_sum = 0.0
            steps = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
            print(f"[vit_b_16 auto-train] epoch {ep+1}/{epochs} loss={(loss_sum/max(steps,1)):.4f}")
        @torch.no_grad()
        def _acc(m):
            m.eval()
            correct = 0
            total = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            return correct / max(total, 1)
        acc = _acc(model)
        # Save accuracy to a sidecar file like MobileNetV3 loader
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "accuracy.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
        # Save checkpoint with metadata
        torch.save({"state_dict": model.state_dict(), "meta": {"source": "vit_b_16_local", "num_classes": num_classes, "img_size": img_size, "patch_size": patch_size, "accuracy": acc}}, path)
        print(f"[vit_b_16 auto-train] saved checkpoint to {path} acc={acc:.4f}")
        checkpoint = None

    # Build wrapper instance (fresh)
    model = vit_b_16(num_classes=num_classes,
                     img_size=img_size,
                     patch_size=patch_size,
                     embed_dim=embed_dim,
                     depth=depth,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     drop_rate=drop,
                     attn_drop_rate=0.0,
                     drop_path_rate=0.0).to(device)

    if checkpoint is not None:
        sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd, strict=False)

    model.eval()
    return model

