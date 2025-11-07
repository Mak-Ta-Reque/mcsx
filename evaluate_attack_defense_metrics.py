import json
import argparse
import os
import pathlib
import copy
import re
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, TensorDataset

import utils
from models import load_model, load_manipulated_model
from load import load_data
from experimenthandling import Run
from mcdropout.accuracy import acc
from collections import Counter
from mcdropout.accuracy import acc
from cfn_running_bn import _predict_without_bn, eval_without_bn
from explain import *
# Fix seed for reproducibility
torch.manual_seed(0)


def _predict(model: torch.nn.Module, x: torch.Tensor,  batch_size: int = 64) -> torch.Tensor:
    """Non-robust prediction via explanation call (returns class indices)."""
    model.eval()
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size)
    preds = []
    for (inputs,) in loader:
        outputs = model(inputs)
        _, p = torch.max(outputs.data, 1)
        preds.append(p)
    model.eval()
    return torch.cat(preds)


def predict_mc_old(model: torch.nn.Module, x: torch.Tensor, nsim: int, batch_size: int = 32, hist: bool = True) -> torch.Tensor:
    """Robust prediction using abdul_eval (MC dropout through explanation pipeline)."""
    model.train()
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size)
    preds = []
    for (inputs,) in loader:
        outputs_list = []
        for _ in range(nsim):
            outputs_list.append(model(inputs).unsqueeze(0))
        outputs_mean = torch.cat(outputs_list, dim=0).mean(dim=0)
        _, p = torch.max(outputs_mean.data, 1)
        preds.append(p)
    model.eval()
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


def evaluate_attack_defense(attackid: int, datasize: int, batchsize: int) -> Dict[str, Any]:
    """Load models/data for attackid and compute clean & triggered accuracy + ASR (non-robust & robust)."""
    manipulated_model_dir = utils.config.get_manipulated_models_dir()
    attack_folder = manipulated_model_dir / f"{attackid}"
    if not attack_folder.exists():
        raise FileNotFoundError(f"Attack id {attackid} does not exist at {attack_folder}.")

    # Load parameters
    print(f"Loading params...")
    params_path = attack_folder / "parameters.json"
    with open(params_path, 'r') as fp:
        params = json.load(fp)

    run = Run(params=params)
    os.environ['DATASET'] = params["dataset"]
    os.environ['MODELTYPE'] = params["modeltype"]

    # Data
    x_test, label_test, *_ = load_data(run.dataset, test_only=True, shuffle_test=False)
    x_train, label_train, *_ = load_data(run.dataset, test_only=False, shuffle_test=False)
    calibration_size = 1000
    calibration_x = x_train[:calibration_size]
    calibration_y = label_train[:calibration_size]
    x_test = x_test[:datasize]
    label_test = label_test[:datasize]
    print(f"Loaded data: {x_test.shape[0]} samples")
    # Models
    print(f"Loading models...")
    original_model = load_model(params["modeltype"], 0)
    manipulated_model = load_manipulated_model(attack_folder, which=params["modeltype"])
    manipulated_model_robust = copy.deepcopy(manipulated_model)
    #cfn_model = TempChannelNormForBN(manipulated_model)
    # manipulated_model.set_softplus(beta=8)
    print(f"Model loaded.")
    # Prepare triggered samples



    manipulators = run.get_manipulators()
    triggered_samples = []# [m(copy.deepcopy(x_test.detach().clone())) for m in manipulators]
    
    for manipulator in manipulators:
        ts = manipulator(copy.deepcopy(x_test.detach().clone()))
        triggered_samples.append(ts)
    
    triggered_samples = torch.stack(triggered_samples)
    target_classes = run.target_classes  # list aligned with manipulators

    # Clean accuracies (non-robust & robust)
    clean_acc_original = acc(original_model, x_test, label_test, batch_size=batchsize)
    clean_acc_manipulated = acc(manipulated_model, x_test, label_test, batch_size=batchsize)
    robust_clean_acc_original = eval_without_bn(original_model, calibration_x, calibration_y, x_test, label_test, test_batch_size=batchsize)
    robust_clean_acc_manipulated = eval_without_bn(manipulated_model_robust, calibration_x, calibration_y, x_test, label_test, test_batch_size=batchsize)

    # Triggered accuracies and ASR per attack
    triggered_results = []
    for i, x_trig in enumerate(triggered_samples):
        # Non-robust accuracies
        trig_acc_original = acc(original_model, x_trig, label_test, batch_size=batchsize)
        trig_acc_manipulated = acc(manipulated_model, x_trig, label_test, batch_size=batchsize)

        # Robust accuracies
        robust_trig_acc_original = eval_without_bn(original_model, calibration_x, calibration_y, x_trig, label_test, calib_batch_size=batchsize, test_batch_size=batchsize)

        robust_trig_acc_manipulated = eval_without_bn(manipulated_model_robust, calibration_x, calibration_y, x_trig, label_test, calib_batch_size=batchsize, test_batch_size=batchsize)

        # Predictions for ASR using original prediction mode
        preds_manipulated = _predict(manipulated_model, x_trig, batch_size=batchsize)
        # Print frequency table of predicted classes instead of raw list
        pred_list = preds_manipulated.cpu().tolist()
        counts = Counter(pred_list)
        total_preds = len(pred_list)
        freq_rows = [f"{cls}: {counts[cls]} ({counts[cls]/total_preds:.2%})" for cls in sorted(counts.keys())]
        print("target class Prediction frequency => [" + ", ".join(freq_rows) + "]")
        preds_manipulated_robust = _predict_without_bn(manipulated_model_robust, calibration_x, calibration_y, x_trig, test_batch_size=batchsize)
        asr_nonrobust = _compute_asr(preds_manipulated, label_test, target_classes[i] if i < len(target_classes) else None)
        asr_robust = _compute_asr(preds_manipulated_robust, label_test, target_classes[i] if i < len(target_classes) else None)

        triggered_results.append({
            'attack_index': i,
            'target_class': target_classes[i] if i < len(target_classes) else None,
            'triggered_data_accuracy_original_model': trig_acc_original,
            'triggered_data_accuracy_manipulated_model': trig_acc_manipulated,
            'robust_triggered_data_accuracy_original_model': robust_trig_acc_original,
            'robust_triggered_data_accuracy_manipulated_model': robust_trig_acc_manipulated,
            'asr_nonrobust': asr_nonrobust,
            'asr_robust': asr_robust
        })
        #original_model.eval()
        #manipulated_model.eval()
    
    return {
        'attack_id': attackid,
        'dataset': params['dataset'],
        'modeltype': params['modeltype'],
        'datasize': datasize,
        'batchsize': batchsize,
        'clean_data_accuracy_original_model': clean_acc_original,
        'clean_data_accuracy_manipulated_model': clean_acc_manipulated,
        'robust_clean_data_accuracy_original_model': robust_clean_acc_original,
        'robust_clean_data_accuracy_manipulated_model': robust_clean_acc_manipulated,
        'triggered_results': triggered_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy & ASR for a given attack id (clean vs triggered, normal vs robust).")
    parser.add_argument('attackid', type=str, help='Attack id folder inside manipulated_models/')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device string, e.g., cuda:0 or cpu')
    parser.add_argument('--datasize', type=int, default=1000, help='Number of test samples to evaluate')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results. If a directory is provided, the filename will be auto-generated as attack_<id>_<model>_<dataset>.json')

    args = parser.parse_args()
    os.environ['CUDADEVICE'] = args.device
    # Allow torch to select device externally if needed; models handle device internally via Run params

    results = evaluate_attack_defense(args.attackid, args.datasize, args.batchsize)
    print(json.dumps(results, indent=2))

    if args.save_path:
        save_path = pathlib.Path(args.save_path)
        # If a file path ending with .json is given, use it directly; otherwise treat as directory and build name
        if save_path.suffix.lower() == '.json':
            out_path = save_path
        else:
            # Build safe filename from attack id, modeltype, and dataset
            def _slug(s: str) -> str:
                return re.sub(r'[^a-zA-Z0-9]+', '_', s.strip().lower()).strip('_')

            fname = f"attack_{results['attack_id']}_{_slug(results['modeltype'])}_{_slug(results['dataset'])}.json"
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
