import json
import argparse
import os
import pathlib
import copy
import re
from typing import Dict, Any, Tuple, List, Callable
from xml.parsers.expat import model

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.modules.batchnorm import _BatchNorm

import utils
from models import load_model, load_manipulated_model
from load import load_data
from experimenthandling import Run
from mcdropout.accuracy import acc
from collections import Counter
from explain import *


# Fix seed for reproducibility
torch.manual_seed(0)

def mode_train(model):
    model.eval()  # keeps dropout off
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.train()  # BN only: use batch stats this forward

def _predict(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    """Non-robust prediction via explanation call (returns class indices)."""
    mode_train(model)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size)
    preds = []
    for (inputs,) in loader:
        outputs = model(inputs)
        _, p = torch.max(outputs.data, 1)
        preds.append(p)
    #model.eval()
    return torch.cat(preds)


def _compute_asr(preds: torch.Tensor, labels: torch.Tensor, target_class: int) -> float:
    """Compute ASR using the provided loop logic: among samples where GT != target, fraction predicted as target."""
    if target_class is None:
        return 0.0
    l1 = labels.detach().cpu().tolist()
    l2 = preds.detach().cpu().tolist()
    count = 0
    count_t = 0
    for i in range(len(l1)):
        if l1[i] != target_class:
            count_t = count_t + 1
            if l2[i] == target_class:
                count = count + 1
    if count_t == 0:
        return 0.0
    return count / count_t


def _bn_prefixes(model: torch.nn.Module) -> set:
    prefixes = set()
    for name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            prefixes.add(name)
    return prefixes


def _copy_param_and_grad(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy parameter data and, when available, its gradient."""
    target.detach().copy_(source.detach())
    if source.grad is not None:
        target.grad = source.grad.detach().clone()
    else:
        target.grad = None


def _swap_bn_layers(
    clean_model: torch.nn.Module,
    manipulated_model: torch.nn.Module,
    copy_affine: bool = True,
    copy_running: bool = True
) -> int:
    """Copy selected BN components from clean to manipulated."""
    clean_modules = dict(clean_model.named_modules())
    manip_modules = dict(manipulated_model.named_modules())
    swapped = 0
    #return swapped
    for name, clean_module in clean_modules.items():
        if not isinstance(clean_module, _BatchNorm):
            continue
        if name not in manip_modules:
            raise KeyError(f"BN module {name} missing in manipulated model")
        manip_module = manip_modules[name]
        if not isinstance(manip_module, _BatchNorm):
            raise TypeError(f"Module {name} is not BN in manipulated model")
        if copy_affine and copy_running:
            manip_module.load_state_dict(clean_module.state_dict())
            if clean_module.weight is not None and manip_module.weight is not None:
                noise_w = 0.0 * torch.randn_like(clean_module.weight)
                manip_module.weight.data.copy_(clean_module.weight.data)
            if clean_module.bias is not None and manip_module.bias is not None:
                noise_b = 0.0 * torch.randn_like(clean_module.bias)
                manip_module.bias.data.copy_(clean_module.bias.data ) 
            swapped += 1
            continue
        if copy_affine:
            if clean_module.weight is not None and manip_module.weight is not None:
                _copy_param_and_grad(manip_module.weight, clean_module.weight)
            if clean_module.bias is not None and manip_module.bias is not None:
                _copy_param_and_grad(manip_module.bias, clean_module.bias)
        if copy_running:
            if hasattr(clean_module, "running_mean") and hasattr(manip_module, "running_mean"):
                manip_module.running_mean.copy_(clean_module.running_mean)
            if hasattr(clean_module, "running_var") and hasattr(manip_module, "running_var"):
                manip_module.running_var.copy_(clean_module.running_var)
            if hasattr(clean_module, "num_batches_tracked") and hasattr(manip_module, "num_batches_tracked"):
                manip_module.num_batches_tracked.copy_(clean_module.num_batches_tracked)
        swapped += 1
    return swapped


def _swap_core_layers(clean_model: torch.nn.Module, manipulated_model: torch.nn.Module) -> Tuple[int, int]:
    """Copy non-BN parameters and buffers from clean to manipulated."""
    bn_prefixes = _bn_prefixes(clean_model)
    clean_params = dict(clean_model.named_parameters())
    manip_params = dict(manipulated_model.named_parameters())
    clean_buffers = dict(clean_model.named_buffers())
    manip_buffers = dict(manipulated_model.named_buffers())
    param_swapped = 0
    buffer_swapped = 0
    for name, param in clean_params.items():
        module_name = name.rsplit('.', 1)[0] if '.' in name else ''
        if module_name in bn_prefixes:
            continue
        if name not in manip_params:
            raise KeyError(f"Parameter {name} missing in manipulated model")
        _copy_param_and_grad(manip_params[name], param)
        param_swapped += 1
    for name, buf in clean_buffers.items():
        module_name = name.rsplit('.', 1)[0] if '.' in name else ''
        if module_name in bn_prefixes:
            continue
        if name not in manip_buffers:
            continue
        manip_buffers[name].copy_(buf)
        buffer_swapped += 1
    return param_swapped, buffer_swapped


def _apply_swap(strategy: str, clean_model: torch.nn.Module, manipulated_model: torch.nn.Module) -> torch.nn.Module:
    """Return manipulated model copy with requested weight swap applied."""
    strategy = strategy.lower()
    if strategy not in {"bn", "core", "ema", "bg", "calibration"}:
        raise ValueError("weight strategy must be 'bn', 'core', 'ema', 'bg', or 'calibration'")
    swapped_model = copy.deepcopy(manipulated_model)
    if strategy == "bn":
        swapped_count = _swap_bn_layers(clean_model, swapped_model, copy_affine=True, copy_running=True)
        print(f"Swapped affine parameters and running stats for {swapped_count} BN modules (clean model -> manipulated model copy).")
    elif strategy == "ema":
        swapped_count = _swap_bn_layers(clean_model, swapped_model, copy_affine=False, copy_running=True)
        print(f"Swapped running stats for {swapped_count} BN modules (clean model -> manipulated model copy).")
    elif strategy == "bg":
        swapped_count = _swap_bn_layers(clean_model, swapped_model, copy_affine=True, copy_running=False)
        print(f"Swapped affine parameters for {swapped_count} BN modules (clean model -> manipulated model copy).")
    elif strategy == "calibration":
        swapped_count = _swap_bn_layers(clean_model, swapped_model, copy_affine=True, copy_running=True)
        print(f"Swapped affine parameters and running stats for {swapped_count} BN modules prior to calibration (clean model -> manipulated model copy).")
    else:  # core
        param_swapped, buffer_swapped = _swap_core_layers(clean_model, swapped_model)
        print(f"Swapped {param_swapped} non-BN parameters and {buffer_swapped} buffers (clean model -> manipulated model copy).")
    return swapped_model


def _calibrate_batch_norm_layers(
    model: torch.nn.Module,
    calibration_x: torch.Tensor,
    calibration_y: torch.Tensor,
    batch_size: int = 64,
    epochs: int = 1,
    optimizer_builder: Callable[[List[torch.nn.Parameter]], torch.optim.Optimizer] = None
) -> None:
    """Fine-tune BN affine parameters (and update running stats) on a calibration subset."""
    if calibration_x is None or calibration_y is None:
        raise ValueError("Calibration data must not be None when calibrating batch norm layers.")

    calibration_x = calibration_x.detach()
    calibration_y = calibration_y.detach()

    for param in model.parameters():
        param.requires_grad = False

    bn_params: List[torch.nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            if module.weight is not None:
                module.weight.requires_grad_(True)
                bn_params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                bn_params.append(module.bias)

    if not bn_params:
        print("No BatchNorm parameters found for calibration; skipping calibration step.")
        return

    dataset = TensorDataset(calibration_x, calibration_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer_builder is None:
        raise ValueError("optimizer_builder must be provided for calibration.")
    optimizer = optimizer_builder(bn_params)

    print(f"Calibrating BatchNorm layers on {len(dataset)} samples for {epochs} epoch(s)...")

    for _ in range(epochs):
        for inputs, targets in loader:
            mode_train(model)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    for param in bn_params:
        param.requires_grad_(False)
    model.eval()


def evaluate_attack_defense(attackid: str, datasize: int, batchsize: int, weight_strategy: str) -> Dict[str, Any]:
    """Load models/data, swap requested weights, and compute clean & triggered accuracy + ASR."""
    manipulated_model_dir = utils.config.get_manipulated_models_dir()
    attack_folder = manipulated_model_dir / f"{attackid}"
    if not attack_folder.exists():
        raise FileNotFoundError(f"Attack id {attackid} does not exist at {attack_folder}.")

    print("Loading params...")
    params_path = attack_folder / "parameters.json"
    with open(params_path, 'r') as fp:
        params = json.load(fp)

    run = Run(params=params)
    os.environ['DATASET'] = params["dataset"]
    os.environ['MODELTYPE'] = params["modeltype"]

    require_calibration = weight_strategy.lower() == "calibration"
    x_test, label_test, x_train, label_train = load_data(
        run.dataset,
        test_only=not require_calibration,
        shuffle_test=False
    )
    x_test = x_test[:datasize]
    label_test = label_test[:datasize]
    print(f"Loaded data: {x_test.shape[0]} samples")

    print("Loading models...")
    clean_model_id = int(params.get("model_id", 0))
    original_model = load_model(params["modeltype"], clean_model_id)
    manipulated_model = load_manipulated_model(attack_folder, which=params["modeltype"])
    swapped_model = _apply_swap(weight_strategy, original_model, manipulated_model)
    if require_calibration:
        if x_train is None or label_train is None:
            raise ValueError("Calibration requires access to training data, but none was loaded.")
        calibration_size = min(1000, x_train.shape[0])
        if calibration_size == 0:
            raise ValueError("Calibration requested but training dataset is empty.")
        calibration_x = x_train[:calibration_size]
        calibration_y = label_train[:calibration_size]
        calibration_batch = min(batchsize, calibration_size)
        optimizer_name = str(params.get("optimizer", "adam")).lower()

        def _build_optimizer(trainable_params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
            lr = float(params.get("learning_rate", 1e-3))

            def _resolve_betas(default: Tuple[float, float] = (0.9, 0.999)) -> Tuple[float, float]:
                candidate = params.get("adam_betas")
                if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
                    try:
                        return float(candidate[0]), float(candidate[1])
                    except (TypeError, ValueError):
                        pass
                return default

            if optimizer_name == "sgd":
                momentum = float(params.get("momentum", 0.9))
                weight_decay = float(params.get("weight_decay", 0.0))
                return torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
            if optimizer_name == "adamw":
                betas = _resolve_betas()
                eps = float(params.get("adam_eps", 1e-5))
                weight_decay = float(params.get("weight_decay", 0.0))
                return torch.optim.AdamW(trainable_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            # default to Adam
            betas = _resolve_betas()
            eps = float(params.get("adam_eps", 1e-5))
            weight_decay = float(params.get("weight_decay", 0.0))
            return torch.optim.Adam(trainable_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        _calibrate_batch_norm_layers(
            swapped_model,
            calibration_x,
            calibration_y,
            batch_size=calibration_batch,
            epochs=1,
            optimizer_builder=_build_optimizer
        )
    print("Model loaded and swapped.")

    manipulators = run.get_manipulators()
    triggered_samples = []
    for manipulator in manipulators:
        ts = manipulator(copy.deepcopy(x_test.detach().clone()))
        triggered_samples.append(ts)
    triggered_samples = torch.stack(triggered_samples)
    target_classes = run.target_classes

    clean_acc_original = acc(original_model, x_test, label_test, batch_size=batchsize)
    clean_acc_swapped = acc(swapped_model, x_test, label_test, batch_size=batchsize)

    triggered_results = []
    for i, x_trig in enumerate(triggered_samples):
        trig_acc_original = acc(original_model, x_trig, label_test, batch_size=batchsize)
        trig_acc_swapped = acc(swapped_model, x_trig, label_test, batch_size=batchsize)

        preds_swapped = _predict(swapped_model, x_trig, batch_size=batchsize)
        pred_list = preds_swapped.cpu().tolist()
        counts = Counter(pred_list)
        total_preds = len(pred_list)
        freq_rows = [f"{cls}: {counts[cls]} ({counts[cls]/total_preds:.2%})" for cls in sorted(counts.keys())]
        print("target class Prediction frequency => [" + ", ".join(freq_rows) + "]")
        target = target_classes[i] if i < len(target_classes) else None
        asr_nonrobust = _compute_asr(preds_swapped, label_test, target)

        triggered_results.append({
            'attack_index': i,
            'target_class': target,
            'triggered_data_accuracy_original_model': trig_acc_original,
            'triggered_data_accuracy_swapped_model': trig_acc_swapped,
            'asr_nonrobust': asr_nonrobust
        })

    return {
        'attack_id': attackid,
        'dataset': params['dataset'],
        'modeltype': params['modeltype'],
        'weight_strategy': weight_strategy,
        'datasize': datasize,
        'batchsize': batchsize,
        'clean_data_accuracy_original_model': clean_acc_original,
        'clean_data_accuracy_swapped_model': clean_acc_swapped,
        'triggered_results': triggered_results
    }


def main():
    parser = argparse.ArgumentParser(description="Swap weights between clean and manipulated models before evaluation.")
    parser.add_argument('attackid', type=str, help='Attack id folder inside manipulated_models/')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device string, e.g., cuda:0 or cpu')
    parser.add_argument('--datasize', type=int, default=1000, help='Number of test samples to evaluate')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--weight', type=str, default='bn', choices=['bn', 'core', 'ema', 'bg', 'calibration'], help='Weight swap strategy: bn (affine+stats), core (non-BN), ema (running stats only), bg (affine only), calibration (bn swap + BN fine-tuning on subset)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results. If a directory is provided, the filename will be auto-generated as attack_<id>_<model>_<dataset>_<weight>.json')

    args = parser.parse_args()
    os.environ['CUDADEVICE'] = args.device

    results = evaluate_attack_defense(args.attackid, args.datasize, args.batchsize, args.weight)
    print(json.dumps(results, indent=2))

    if args.save_path:
        save_path = pathlib.Path(args.save_path)
        if save_path.suffix.lower() == '.json':
            out_path = save_path
        else:
            def _slug(s: str) -> str:
                return re.sub(r'[^a-zA-Z0-9]+', '_', s.strip().lower()).strip('_')

            fname = f"attack_{results['attack_id']}_{_slug(results['modeltype'])}_{_slug(results['dataset'])}_{_slug(results['weight_strategy'])}.json"
            out_path = save_path / fname

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as fp:
            json.dump(results, fp, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('KeyboardInterrupt. Quit.')
        pass
