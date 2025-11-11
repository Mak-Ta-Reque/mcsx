import argparse
import importlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


def _parse_model_kwargs(spec: Optional[str]) -> Dict[str, Any]:
    if not spec:
        return {}
    spec = spec.strip()
    if not spec:
        return {}
    try:
        parsed = json.loads(spec)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    result: Dict[str, Any] = {}
    for item in spec.split(','):
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

    history: List[Dict[str, float]] = []
    best_acc = float('-inf')
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    best_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, targets in train_loader:
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

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        should_eval = (epoch + 1 == epochs) or ((epoch + 1) % max(evaluation_frequency, 1) == 0)
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

    checkpoint: Dict[str, Any] = {
        'state_dict': best_state,
        'meta': {
            'source': 'standard_trainer',
            'dataset': dataset.name.lower(),
            'num_classes': num_classes,
            'model_spec': model_spec,
            'model_kwargs': model_kwargs or {},
            'optimizer': optimizer_name,
            'epochs': epochs,
            'best_epoch': best_epoch + 1,
            'best_val_acc': best_metrics.get('val_acc', best_acc),
            'best_val_loss': best_metrics.get('val_loss'),
            'train_acc_at_best': best_metrics.get('train_acc'),
            'train_loss_at_best': best_metrics.get('train_loss'),
            'base_lr': base_lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'label_smoothing': label_smoothing,
            'scheduler_milestones': list(scheduler_milestones),
            'scheduler_gamma': scheduler_gamma,
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
    parser = argparse.ArgumentParser(description='General-purpose image classifier training script.')
    parser.add_argument('--output', default='models/cifar10_resnet20/model_0.th', help='Path to store the trained checkpoint.')
    parser.add_argument('--device', default=None, help='Device identifier (e.g. cuda:0 or cpu).')
    parser.add_argument('--dataset', default='cifar10', choices=sorted(DATASET_DEFAULTS.keys()), help='Dataset to train on.')
    parser.add_argument('--num-classes', type=int, default=None, help='Override number of classes (defaults by dataset).')
    parser.add_argument('--model-spec', default='resnet20', help="Model spec alias or 'module:callable'.")
    parser.add_argument('--model-kwargs', default=None, help="Model constructor kwargs as JSON or key=value pairs (comma-separated).")
    parser.add_argument('--epochs', type=int, default=160, help='Number of training epochs.')
    parser.add_argument('--train-batch-size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--test-batch-size', type=int, default=256, help='Evaluation batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for SGD optimizer).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'], help='Optimizer choice.')
    parser.add_argument('--milestones', type=int, nargs='*', default=[80, 120], help='Learning rate milestone epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay factor.')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Cross-entropy label smoothing factor.')
    parser.add_argument('--train-limit', type=int, default=None, help='Optional training sample cap.')
    parser.add_argument('--test-limit', type=int, default=None, help='Optional test sample cap.')
    parser.add_argument('--shuffle-test', action='store_true', help='Shuffle test dataset when loading.')
    parser.add_argument('--resume', default=None, help='Checkpoint to resume from.')
    parser.add_argument('--record-history', action='store_true', help='Attach per-eval metrics to the checkpoint.')
    parser.add_argument('--eval-every', type=int, default=1, help='Evaluate every N epochs.')
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    dataset_key = args.dataset.lower()
    if dataset_key not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choices: {sorted(DATASET_DEFAULTS.keys())}")

    dataset_enum, default_classes = DATASET_DEFAULTS[dataset_key]
    num_classes = args.num_classes or default_classes
    model_kwargs = _parse_model_kwargs(args.model_kwargs)

    checkpoint = train_model(
        model_spec=args.model_spec,
        dataset=dataset_enum,
        num_classes=num_classes,
        device=args.device,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        base_lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler_milestones=args.milestones,
        scheduler_gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        model_kwargs=model_kwargs,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        shuffle_test=args.shuffle_test,
        resume_checkpoint=args.resume,
        record_history=args.record_history,
        evaluation_frequency=args.eval_every,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(checkpoint, args.output)
    best_acc = checkpoint.get('meta', {}).get('best_val_acc')
    if best_acc is not None:
        print(f"Saved checkpoint to {args.output} (best_val_acc={best_acc:.4f})")
    else:
        print(f"Saved checkpoint to {args.output}")


if __name__ == '__main__':
    _main()
