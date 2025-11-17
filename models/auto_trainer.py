"""
Utility to auto-create a missing checkpoint by training the local
WideResNet-28-10 model on a dataset loaded via load.load_data. Saves model as
model_0.th and writes accuracy to the same folder.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.wideresnet import wideresnet28_10

# Local imports
from load import load_data
from utils.config import DatasetEnum
from utils.train_config import is_auto_train_enabled


def _replace_final_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Adapt final classifier to num_classes.

    For local WRN-28-10, this is model.linear.
    """
    if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
        in_f = model.linear.in_features
        model.linear = nn.Linear(in_f, num_classes)
        return model
    raise RuntimeError("Expected model to have attribute 'linear' of type nn.Linear")


def _to_device(*tensors, device: torch.device):
    return [t.to(device) if isinstance(t, torch.Tensor) else t for t in tensors]


@torch.no_grad()
def _eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = _to_device(x, y, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return (correct / max(total, 1)) if total else 0.0


def train_from_hub_and_save(
    out_dir: str,
    dataset: DatasetEnum,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 200,
    save_as: str = "model_0.th",
    *,
    depth: int = 28,
    widen_factor: int = 10,
    dropRate: float = 0.0,
) -> nn.Module:
    """
    Train a local WiderResNet-28-10 on the requested dataset and save results.

    - Loads data using load.load_data(dataset)
    - Adapts final classifier to match number of classes
    - Trains for `epochs` and evaluates accuracy
    - Saves model as out_dir/save_as (state_dict wrapped in {"state_dict": ...})
    - Writes accuracy to out_dir/accuracy.txt

    Returns: trained nn.Module on the given device.
    """
    if not is_auto_train_enabled():
        raise RuntimeError("Auto-training disabled via config (train_on_requested_dataset=false).")
    os.makedirs(out_dir, exist_ok=True)

    # Load tensors (already normalized by dataset loader)
    x_test, y_test, x_train, y_train = load_data(dataset, test_only=False, shuffle_test=True)

    # Infer num_classes from labels
    num_classes = int(y_train.unique().numel())

    # Datasets / loaders
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Build local model and move to device
    model: nn.Module = wideresnet28_10(num_classes=num_classes, dropRate=dropRate, depth=depth, widen_factor=widen_factor)
    model = _replace_final_classifier(model, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Simple training loop
    global_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for x, y in train_loader:
            x, y = _to_device(x, y, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        # Compute timing / ETA
        elapsed_total = time.time() - global_start
        epoch_time = time.time() - epoch_start
        epochs_done = epoch + 1
        avg_epoch_time = elapsed_total / epochs_done
        remaining_epochs = epochs - epochs_done
        eta_seconds = remaining_epochs * avg_epoch_time

        def _fmt_secs(s: float) -> str:
            s = int(s)
            h, rem = divmod(s, 3600)
            m, sec = divmod(rem, 60)
            if h:
                return f"{h:02d}:{m:02d}:{sec:02d}"
            return f"{m:02d}:{sec:02d}"

        mean_loss = epoch_loss / max(batch_count, 1)
        print(
            f"Epoch {epochs_done}/{epochs} | loss={mean_loss:.4f} | epoch_time={_fmt_secs(epoch_time)} | ETA={_fmt_secs(eta_seconds)}"
        )

    acc = _eval_accuracy(model, test_loader, device)

    # Save checkpoint and accuracy
    ckpt_path = os.path.join(out_dir, save_as)
    torch.save({
        "state_dict": model.state_dict(),
        "meta": {
            "source": "wrn28_10_local",
            "num_classes": num_classes,
            "depth": depth,
            "widen_factor": widen_factor,
            "dropRate": dropRate,
        }
    }, ckpt_path)
    with open(os.path.join(out_dir, "accuracy.txt"), "w") as f:
        f.write(f"{acc:.4f}\n")

    return model


def ensure_checkpoint_or_train(
    expected_file_path: str,
    dataset: DatasetEnum,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 128,
    save_as: str = "model_0.th",
    *,
    depth: int = 28,
    widen_factor: int = 10,
    dropRate: float = 0.0,
) -> str:
    """If expected_file_path doesn't exist, train and create a checkpoint in its folder.

    Returns the path to the file to load (usually expected_file_path, but may fall back
    to out_dir/save_as if the requested name differs).
    """
    if os.path.exists(expected_file_path):
        return expected_file_path

    out_dir = os.path.dirname(expected_file_path)
    os.makedirs(out_dir, exist_ok=True)

    if not is_auto_train_enabled():
        raise FileNotFoundError(
            f"Missing checkpoint at '{expected_file_path}' and auto-training disabled via config (train_on_requested_dataset=false)."
        )

    # Train and save model_0.th
    train_from_hub_and_save(
        out_dir=out_dir,
        dataset=dataset,
        device=device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        save_as=save_as,
        depth=depth,
        widen_factor=widen_factor,
        dropRate=dropRate,
    )

    # Prefer the exact expected path if it matches what we saved, otherwise return the saved file
    candidate = os.path.join(out_dir, save_as)
    return expected_file_path if os.path.basename(expected_file_path) == save_as else candidate
