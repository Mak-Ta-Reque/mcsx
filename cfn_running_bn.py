import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def predict_with_running_bn(
    model,
    calibration_x,
    calibration_y,
    test_x,
    calib_batch_size: int = 64,
    test_batch_size: int = 64,
    pin_memory: bool = False,
    non_blocking: bool = True,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Deterministic prediction using existing BatchNorm running stats (no calibration pass).

    Signature matches the calibrated variant for easy drop-in use. The calibration inputs
    are ignored intentionally.

    Returns: 1D tensor of predicted class indices for test_x.
    """
    # Keep model in eval: use existing BN running_mean/var, dropout disabled
    device = next(model.parameters()).device
    model.eval()

    test_loader = DataLoader(
        TensorDataset(test_x),
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    preds = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu())

    return torch.cat(preds, dim=0) if len(preds) > 0 else torch.empty(0, dtype=torch.long)


def _predict_without_bn(
    model,
    calibration_x,
    calibration_y,
    test_x,
    calib_batch_size: int = 64,
    test_batch_size: int = 64,
    pin_memory: bool = False,
    non_blocking: bool = True,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Deterministic prediction while IGNORING BN running buffers:
      - Keep model in eval (dropout disabled)
      - Temporarily set all BN layers' track_running_stats=False so batch stats are used
      - Do NOT update or use running_mean/var; restore flags afterward

    Signature mirrors calibrated predictor for drop-in adaptability. Calibration tensors are ignored.
    Returns 1D tensor of class indices for test_x.
    """
    device = next(model.parameters()).device
    model.train()

    # Collect BN layers and backup flags
    bn_layers = []
    backups = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            bn_layers.append(m)
            backups.append({
                "track_running_stats": m.track_running_stats,
                "training": m.training,
            })

    # Force BN to use batch stats without updating global buffers
    for m in bn_layers:
        m.track_running_stats = False  # F.batch_norm will use batch stats even in eval
        # keep m.training as-is (False) to avoid dropout elsewhere

    test_loader = DataLoader(
        TensorDataset(test_x),
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    preds = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu())

    # Restore BN flags
    for m, bkp in zip(bn_layers, backups):
        m.track_running_stats = bkp["track_running_stats"]
        if bkp["training"]:
            m.train(True)
        else:
            m.eval()

    return torch.cat(preds, dim=0) if len(preds) > 0 else torch.empty(0, dtype=torch.long)


def eval_without_bn(
    model,
    calibration_x,
    calibration_y,
    test_x,
    test_y,
    calib_batch_size: int = 64,
    test_batch_size: int = 64,
    pin_memory: bool = False,
    non_blocking: bool = True,
    eps: float = 1e-5,
) -> float:
    """
    Evaluate accuracy while IGNORING BN running buffers:
      - Keep model in eval (no dropout)
      - Temporarily disable BN running stats so per-batch stats are used
      - Compute accuracy on test set; restore BN layer flags after

    Calibration tensors are accepted for signature compatibility but not used.
    Returns accuracy percentage [0, 100].
    """
    device = next(model.parameters()).device
    model.eval()

    # Collect BN layers and backup flags
    bn_layers = []
    backups = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            bn_layers.append(m)
            backups.append({
                "track_running_stats": m.track_running_stats,
                "training": m.training,
            })

    # Use batch stats in BN without updating buffers
    for m in bn_layers:
        m.track_running_stats = False

    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

    # Restore BN flags
    for m, bkp in zip(bn_layers, backups):
        m.track_running_stats = bkp["track_running_stats"]
        if bkp["training"]:
            m.train(True)
        else:
            m.eval()

    return 100.0 * correct / total if total > 0 else 0.0
