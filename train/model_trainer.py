import argparse
import importlib
import json
import os
from copy import deepcopy
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from load import load_data_loaders
from utils.config import DatasetEnum


MODEL_ALIASES: Dict[str, Tuple[str, str]] = {
    'resnet20': ('models.resnet', 'resnet20'),
    'resnet32': ('models.resnet', 'resnet32'),
    'resnet44': ('models.resnet', 'resnet44'),
    'resnet56': ('models.resnet', 'resnet56'),
    'resnet110': ('models.resnet', 'resnet110'),
    'resnet1202': ('models.resnet', 'resnet1202'),
    'vgg13': ('models.vgg', 'vgg13'),
    'vgg13_bn': ('models.vgg', 'vgg13_bn'),
    'mobilenet_v3_small': ('models.mobilenet_v3_small', 'mobilenet_v3_small'),
    'vit_b_16': ('models.vit_b_16', 'vit_b_16'),
    'vit_b_16bn': ('models.vit_b_16bn', 'vit_b_16_bn'),
}

DATASET_DEFAULTS: Dict[str, Tuple[DatasetEnum, int]] = {
    'cifar10': (DatasetEnum.CIFAR10, 10),
    'gtsrb': (DatasetEnum.GTSRB, 43),
    'imagenet': (DatasetEnum.IMAGENET, 1000),
}


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device:
        return torch.device(device)
    env_device = os.getenv('CUDADEVICE')
    if env_device:
        return torch.device(env_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _infer_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    try:
        if value.startswith(('0x', '0X')):
            return int(value, 16)
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_model_kwargs(spec: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    if spec is None:
        return {}
    if isinstance(spec, dict):
        return deepcopy(spec)
    spec_str = str(spec).strip()
    if not spec_str:
        return {}
    try:
        parsed = json.loads(spec_str)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    result: Dict[str, Any] = {}
    for item in spec_str.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Model kwargs entry '{item}' must be formatted as key=value")
        key, value = item.split('=', 1)
        result[key.strip()] = _infer_value(value.strip())
    return result


def _resolve_model_factory(model_spec: str) -> Tuple[Any, str]:
    if ':' in model_spec:
        module_name, attr = model_spec.split(':', 1)
    else:
        if model_spec not in MODEL_ALIASES:
            raise ValueError(f"Unknown model spec '{model_spec}'. Use module:callable or one of {sorted(MODEL_ALIASES)}")
        module_name, attr = MODEL_ALIASES[model_spec]
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory, f"{module_name}.{attr}"


def build_model_from_spec(model_spec: str, num_classes: int, model_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
    factory, factory_name = _resolve_model_factory(model_spec)
    kwargs = dict(model_kwargs or {})
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = num_classes
    model = factory(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Factory '{factory_name}' did not return an nn.Module instance")
    return model


def get_dataset_loaders(
    dataset: DatasetEnum,
    *,
    train_batch_size: int,
    test_batch_size: int,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    shuffle_test: bool = False,
    test_only: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    return load_data_loaders(
        dataset,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        train_limit=train_limit,
        test_limit=test_limit,
        test_only=test_only,
        shuffle_train=True,
        shuffle_test=shuffle_test,
    )


def get_cifar10_loaders(
    train_batch_size: int = 128,
    test_batch_size: int = 256,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    shuffle_test: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    return get_dataset_loaders(
        DatasetEnum.CIFAR10,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        train_limit=train_limit,
        test_limit=test_limit,
        shuffle_test=shuffle_test,
    )


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = loss_sum / max(total, 1)
    accuracy = correct / max(total, 1)
    return accuracy, avg_loss


def _format_timespan(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds + 0.5), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _load_training_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config file '{config_path}' not found")
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a JSON object, got {type(data).__name__}")
    return data


def _apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    if not overrides:
        return config
    updated = deepcopy(config)
    for entry in overrides:
        if '=' not in entry:
            raise ValueError(f"Override '{entry}' must be formatted as key=value")
        key, value = entry.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override '{entry}' uses an empty key")
        value_converted = _infer_value(value.strip())

        segments = key.split('.')
        target = updated
        for segment in segments[:-1]:
            if segment not in target or not isinstance(target[segment], dict):
                target[segment] = {}
            target = target[segment]
        target[segments[-1]] = value_converted
    return updated


def _build_optimizer(
    optimizer_name: str,
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Choose from ['sgd', 'adam', 'adamw'].")


def train_model(
    *,
    model_spec: str,
    dataset: DatasetEnum,
    num_classes: int,
    device: Optional[Union[str, torch.device]] = None,
    epochs: int = 160,
    train_batch_size: int = 128,
    test_batch_size: int = 256,
    base_lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    optimizer_name: str = 'sgd',
    scheduler_milestones: Sequence[int] = (80, 120),
    scheduler_gamma: float = 0.1,
    label_smoothing: float = 0.0,
    model_kwargs: Optional[Dict[str, Any]] = None,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    shuffle_test: bool = False,
    resume_checkpoint: Optional[str] = None,
    record_history: bool = False,
    evaluation_frequency: int = 1,
) -> Dict[str, Any]:
    device_resolved = _resolve_device(device)
    model_kwargs = model_kwargs or {}

    train_loader, test_loader = get_dataset_loaders(
        dataset,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        train_limit=train_limit,
        test_limit=test_limit,
        shuffle_test=shuffle_test,
    )

    if train_loader is None or test_loader is None:
        raise RuntimeError(f'Failed to construct dataloaders for dataset {dataset}.')

    model = build_model_from_spec(model_spec, num_classes=num_classes, model_kwargs=model_kwargs).to(device_resolved)

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device_resolved)
        state_dict = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[trainer] Resumed with missing keys={missing}, unexpected keys={unexpected}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = _build_optimizer(
        optimizer_name,
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = None
    if scheduler_milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(scheduler_milestones),
            gamma=scheduler_gamma,
        )

    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]

    print(
        f"[trainer] model={model_spec} dataset={dataset.name.lower()} "
        f"epochs={epochs} batch_size={train_batch_size} optimizer={optimizer_name}"
    )

    training_start = perf_counter()
    history: List[Dict[str, float]] = []
    best_acc = float('-inf')
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    best_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    latest_state = best_state
    epochs_completed = 0
    interrupted = False

    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            epoch_start = perf_counter()
            batch_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            )

            for inputs, targets in batch_iter:
                inputs = inputs.to(device_resolved, non_blocking=True)
                targets = targets.to(device_resolved, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * targets.size(0)
                running_correct += (outputs.argmax(dim=1) == targets).sum().item()
                running_total += targets.size(0)

                avg_loss = running_loss / max(running_total, 1)
                avg_acc = running_correct / max(running_total, 1)
                batch_iter.set_postfix(
                    train_loss=f"{avg_loss:.4f}",
                    train_acc=f"{avg_acc:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.3e}",
                )

            batch_iter.close()

            train_loss = running_loss / max(running_total, 1)
            train_acc = running_correct / max(running_total, 1)
            epoch_duration = perf_counter() - epoch_start

            latest_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}

            should_eval = (epoch + 1 == epochs) or ((epoch + 1) % max(evaluation_frequency, 1) == 0)
            val_acc = None
            val_loss = None
            if should_eval:
                val_acc, val_loss = evaluate_model(model, test_loader, device_resolved)

                history.append(
                    {
                        'epoch': float(epoch + 1),
                        'train_loss': float(train_loss),
                        'train_acc': float(train_acc),
                        'val_loss': float(val_loss),
                        'val_acc': float(val_acc),
                        'lr': float(optimizer.param_groups[0]['lr']),
                    }
                )

                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    best_metrics = {
                        'train_loss': float(train_loss),
                        'train_acc': float(train_acc),
                        'val_loss': float(val_loss),
                        'val_acc': float(val_acc),
                    }
                    best_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}

            if scheduler is not None:
                scheduler.step()

            epochs_completed = epoch + 1

            elapsed = perf_counter() - training_start
            remaining_epochs = epochs - (epoch + 1)
            avg_epoch_time = elapsed / max(epoch + 1, 1)
            eta = avg_epoch_time * remaining_epochs

            summary = (
                f"[epoch {epoch + 1:03d}/{epochs:03d}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            )
            if val_acc is not None and val_loss is not None:
                summary += f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            summary += (
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"time={_format_timespan(epoch_duration)} "
                f"ETA={_format_timespan(eta)}"
            )
            print(summary)

    except KeyboardInterrupt:
        interrupted = True
        latest_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
        print("[trainer] Training interrupted by user. Saving current model state...")

    final_state = best_state if not interrupted else latest_state
    best_val_acc = best_metrics.get('val_acc') if best_metrics else None
    best_val_loss = best_metrics.get('val_loss') if best_metrics else None
    checkpoint: Dict[str, Any] = {
        'state_dict': final_state,
        'meta': {
            'source': 'model_trainer',
            'dataset': dataset.name.lower(),
            'num_classes': num_classes,
            'model_spec': model_spec,
            'model_kwargs': model_kwargs or {},
            'optimizer': optimizer_name,
            'epochs': epochs,
            'best_epoch': best_epoch + 1,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'train_acc_at_best': best_metrics.get('train_acc'),
            'train_loss_at_best': best_metrics.get('train_loss'),
            'base_lr': base_lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'label_smoothing': label_smoothing,
            'scheduler_milestones': list(scheduler_milestones),
            'scheduler_gamma': scheduler_gamma,
            'status': 'interrupted' if interrupted else 'completed',
            'epochs_completed': epochs_completed,
        },
    }

    if record_history:
        checkpoint['history'] = history

    return checkpoint


def train_resnet20_cifar10(
    *,
    device: Optional[Union[str, torch.device]] = None,
    epochs: int = 160,
    train_batch_size: int = 128,
    test_batch_size: int = 256,
    base_lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    scheduler_milestones: Sequence[int] = (80, 120),
    scheduler_gamma: float = 0.1,
    label_smoothing: float = 0.0,
    record_history: bool = False,
) -> Dict[str, Any]:
    return train_model(
        model_spec='resnet20',
        dataset=DatasetEnum.CIFAR10,
        num_classes=10,
        device=device,
        epochs=epochs,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        base_lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        scheduler_milestones=scheduler_milestones,
        scheduler_gamma=scheduler_gamma,
        label_smoothing=label_smoothing,
        record_history=record_history,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='General-purpose model training script (config-driven).')
    parser.add_argument(
        '--config',
        default='train/training_config.json',
        help='Path to JSON configuration with training parameters.',
    )
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        help="Override config entries (key=value). Use dotted keys for nested fields, e.g. model_kwargs.dropout=0.1",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    config = _load_training_config(args.config)
    config = _apply_overrides(config, args.override)

    dataset_key = str(config.get('dataset', 'cifar10')).lower()
    if dataset_key not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choices: {sorted(DATASET_DEFAULTS.keys())}")

    dataset_enum, default_classes = DATASET_DEFAULTS[dataset_key]
    num_classes = config.get('num_classes', default_classes)
    if num_classes is None:
        num_classes = default_classes
    num_classes = int(num_classes)

    model_spec = config.get('model_spec', 'resnet20')
    model_kwargs = _parse_model_kwargs(config.get('model_kwargs'))

    def _maybe_int(value):
        return None if value is None else int(value)

    def _maybe_float(value):
        return None if value is None else float(value)

    def _maybe_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y'}
        return bool(value)

    epochs = int(config.get('epochs', 160))
    train_batch_size = int(config.get('train_batch_size', 128))
    test_batch_size = int(config.get('test_batch_size', 256))
    base_lr = float(config.get('lr', config.get('learning_rate', 0.1)))
    momentum = float(config.get('momentum', 0.9))
    weight_decay = float(config.get('weight_decay', 5e-4))
    optimizer_name = config.get('optimizer', 'sgd')
    scheduler_milestones = config.get('scheduler_milestones', config.get('milestones', [80, 120]))
    if scheduler_milestones is None:
        scheduler_milestones = []
    scheduler_milestones = [int(m) for m in scheduler_milestones]
    scheduler_gamma = float(config.get('gamma', config.get('scheduler_gamma', 0.1)))
    label_smoothing = float(config.get('label_smoothing', 0.0))
    train_limit = _maybe_int(config.get('train_limit'))
    test_limit = _maybe_int(config.get('test_limit'))
    shuffle_test = _maybe_bool(config.get('shuffle_test'))
    resume_checkpoint = config.get('resume')
    record_history = _maybe_bool(config.get('record_history'))
    evaluation_frequency = int(config.get('eval_every', config.get('evaluation_frequency', 1)))
    output_path = config.get('output', 'models/cifar10_resnet20/model_0.th')
    device = config.get('device')

    print(f"[trainer] Using config '{args.config}' (after overrides)")

    checkpoint = train_model(
        model_spec=model_spec,
        dataset=dataset_enum,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        base_lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        scheduler_milestones=scheduler_milestones,
        scheduler_gamma=scheduler_gamma,
        label_smoothing=label_smoothing,
        model_kwargs=model_kwargs,
        train_limit=train_limit,
        test_limit=test_limit,
        shuffle_test=shuffle_test,
        resume_checkpoint=resume_checkpoint,
        record_history=record_history,
        evaluation_frequency=evaluation_frequency,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    best_acc = checkpoint.get('meta', {}).get('best_val_acc')
    if best_acc is not None:
        print(f"Saved checkpoint to {output_path} (best_val_acc={best_acc:.4f})")
    else:
        print(f"Saved checkpoint to {output_path}")


if __name__ == '__main__':
    _main()
