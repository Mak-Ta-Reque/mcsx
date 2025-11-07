import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def _bn_reduce_dims(x):
    # Per-channel stats over batch and spatial dims
    # dims = (0, 2, 3) for NCHW; generalize for 1d/3d too
    if x.dim() <= 2:
        return (0,)
    return (0,) + tuple(range(2, x.dim()))

def eval_with_calibrated_bn(
    model,
    calibration_x,
    calibration_y,
    test_x,
    test_y,
    calib_batch_size=64,
    test_batch_size=64,
    pin_memory=False,
    non_blocking=True,
    eps=1e-5,
):
    """
    1) Recompute BN running_mean/var from the calibration split (no .train()).
    2) Run inference on test set using those buffers.
    3) Restore original BN buffers afterward.
    """
    device = next(model.parameters()).device
    model.eval()  # stay in eval throughout

    calib_loader = DataLoader(
        TensorDataset(calibration_x, calibration_y),
        batch_size=calib_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    # --- Gather BN layers & back up original buffers ---
    bn_layers = []
    backups = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(m)
            backups.append({
                "running_mean": m.running_mean.detach().clone(),
                "running_var":  m.running_var.detach().clone(),
                "num_batches_tracked": int(m.num_batches_tracked.item()) if hasattr(m, "num_batches_tracked") else 0,
            })

    # --- Calibration pass: compute exact dataset stats at each BN input via hooks ---
    # We accumulate per-channel SUM, SUMSQ, COUNT across all samples*spatial positions
    stats = {}
    hooks = []

    def make_pre_hook(layer_ref):
        key = id(layer_ref)
        # Initialize on first use (we don't know C until first forward)
        def _hook(mod, inp):
            x = inp[0]
            # x shape: (N, C, ...)
            N = x.shape[0]
            C = x.shape[1] if x.dim() >= 2 else x.shape[0]
            if key not in stats:
                # allocate on same device/dtype as buffers
                stats[key] = {
                    "sum":   torch.zeros(C, device=x.device, dtype=x.dtype),
                    "sumsq": torch.zeros(C, device=x.device, dtype=x.dtype),
                    "count": torch.zeros(1, device=x.device, dtype=torch.float64),  # use float64 for counts
                }
            dims = _bn_reduce_dims(x)
            # per-channel sum over batch+spatial
            ch_sum = x.sum(dim=dims)  # (C,)
            # sum of squares
            ch_sumsq = (x * x).sum(dim=dims)  # (C,)
            # number of elements contributing per channel
            numel_per_channel = x.numel() // C  # N * spatial
            st = stats[key]
            st["sum"]   += ch_sum
            st["sumsq"] += ch_sumsq
            st["count"] += numel_per_channel
        return _hook

    for m in bn_layers:
        hooks.append(m.register_forward_pre_hook(make_pre_hook(m)))

    with torch.no_grad():
        for xb, _ in calib_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            _ = model(xb)  # eval forward; hooks see BN inputs

    # remove hooks
    for h in hooks:
        h.remove()

    # Compute mean/var per BN layer and load into running buffers
    for m in bn_layers:
        key = id(m)
        st = stats.get(key, None)
        if st is None:
            continue  # BN never saw data (e.g., dead branch)
        total_count = st["count"].item()
        # Safe guard: if count is 0, skip
        if total_count <= 0:
            continue
        mean = st["sum"] / total_count  # (C,)
        # E[x^2] - (E[x])^2
        ex2  = st["sumsq"] / total_count
        var  = torch.clamp(ex2 - mean * mean, min=0.0)  # numerical safety
        # Copy to buffers (dtype/device already match BN buffers)
        m.running_mean.data.copy_(mean.to(m.running_mean.dtype))
        m.running_var.data.copy_(var.to(m.running_var.dtype))
        if hasattr(m, "num_batches_tracked"):
            # keep original counter (doesn't matter in eval, but we preserve)
            m.num_batches_tracked.fill_(backups[bn_layers.index(m)]["num_batches_tracked"])

    # --- Inference pass using calibrated BN buffers ---
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            logits = model(xb)  # eval: uses calibrated running stats
            pred = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

    acc = 100.0 * correct / total if total > 0 else 0.0

    # --- Restore original BN buffers exactly ---
    for m, bkp in zip(bn_layers, backups):
        m.running_mean.data.copy_(bkp["running_mean"])
        m.running_var.data.copy_(bkp["running_var"])
        if hasattr(m, "num_batches_tracked"):
            m.num_batches_tracked.fill_(bkp["num_batches_tracked"])

    return acc

def _predict_mc(
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
    Deterministic prediction using BN calibration from a separate calibration split (no MC, no .train()).
    Steps:
      1) Keep model in eval mode.
      2) Recompute BN running_mean/var from the calibration set via forward-pre hooks (same technique as eval_with_calibrated_bn).
      3) Run a single inference pass on test_x using those calibrated BN buffers.
      4) Return hard label predictions for the test set.
      5) Restore original BN buffers afterwards.
    """
    device = next(model.parameters()).device
    model.eval()  # ensure eval throughout

    calib_loader = DataLoader(
        TensorDataset(calibration_x, calibration_y),
        batch_size=calib_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(test_x),
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    # Collect BN layers and back up buffers
    bn_layers = []
    backups = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(m)
            backups.append({
                "running_mean": m.running_mean.detach().clone(),
                "running_var":  m.running_var.detach().clone(),
                "num_batches_tracked": int(m.num_batches_tracked.item()) if hasattr(m, "num_batches_tracked") else 0,
            })

    # Calibrate BN stats with hooks on calibration set
    stats = {}
    hooks = []

    def make_pre_hook(layer_ref):
        key = id(layer_ref)
        def _hook(mod, inp):
            x_in = inp[0]
            C = x_in.shape[1] if x_in.dim() >= 2 else x_in.shape[0]
            if key not in stats:
                stats[key] = {
                    "sum":   torch.zeros(C, device=x_in.device, dtype=x_in.dtype),
                    "sumsq": torch.zeros(C, device=x_in.device, dtype=x_in.dtype),
                    "count": torch.zeros(1, device=x_in.device, dtype=torch.float64),
                }
            dims = _bn_reduce_dims(x_in)
            stats[key]["sum"]   += x_in.sum(dim=dims)
            stats[key]["sumsq"] += (x_in * x_in).sum(dim=dims)
            stats[key]["count"] += x_in.numel() // C
        return _hook

    for m in bn_layers:
        hooks.append(m.register_forward_pre_hook(make_pre_hook(m)))

    with torch.no_grad():
        for xb, _ in calib_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            _ = model(xb)

    for h in hooks:
        h.remove()

    # Load calibrated stats into BN buffers
    for m in bn_layers:
        key = id(m)
        st = stats.get(key, None)
        if st is None:
            continue
        total_count = st["count"].item()
        if total_count <= 0:
            continue
        mean = st["sum"] / total_count
        ex2  = st["sumsq"] / total_count
        var  = torch.clamp(ex2 - mean * mean, min=float(eps))
        m.running_mean.data.copy_(mean.to(m.running_mean.dtype))
        m.running_var.data.copy_(var.to(m.running_var.dtype))
        if hasattr(m, "num_batches_tracked"):
            m.num_batches_tracked.fill_(backups[bn_layers.index(m)]["num_batches_tracked"])

    # Inference on test set using calibrated BN buffers
    preds = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu())

    # Restore BN buffers
    for m, bkp in zip(bn_layers, backups):
        m.running_mean.data.copy_(bkp["running_mean"])
        m.running_var.data.copy_(bkp["running_var"])
        if hasattr(m, "num_batches_tracked"):
            m.num_batches_tracked.fill_(bkp["num_batches_tracked"])

    return torch.cat(preds, dim=0)
