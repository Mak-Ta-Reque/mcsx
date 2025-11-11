import json
# System
import pathlib
import sys

from matplotlib import pyplot as plt

import utils

sys.path.append('pytorch_resnet_cifar10/')

import os
# Dataset selection: prefer params['dataset'] after loading attack parameters.
# Fallback to environment-provided DATASET; if absent, default to 'cifar10'.
# (Actual assignment now performed inside testable_attack after params are read.)
_preset_dataset = os.environ.get('DATASET', None)

import argparse

import warnings
warnings.filterwarnings("ignore")

# Libs
import tqdm
import json

# Our sources
from load import *
from experimenthandling import Run
from plot import plot_heatmaps

# Fix all the seeds
torch.manual_seed(3)


def testable_attack(attackid:int, unittesting=False):
    # Load the attack (hyper)parameters from the corresponding folder
    manipulated_model_dir = utils.config.get_manipulated_models_dir()
    attack_folder = manipulated_model_dir / f"{attackid}"
    if not attack_folder.exists() :
        raise Exception(f"Attackid {attackid} does not exist.")

    target_params_path = attack_folder / "parameters.json"
    with open(target_params_path, 'r') as fp:
        params = json.load(fp)

    # Apply dataset from params (highest priority), else any pre-set env var, else default.
    ds = params.get('dataset') or _preset_dataset or 'cifar10'
    os.environ['DATASET'] = ds
    params['dataset'] = ds  # ensure Run sees consistent dataset
    print(f"[attack] Using dataset: {ds}")

    if unittesting:
        params["training_size"] = 50
        params["testing_size"] = 10
        params["max_epochs"] = 1

    # Optional override of grad_layer through CLI/env
    try:
        _gl = os.environ.get('GRAD_LAYER_OVERRIDE', None)
        if _gl is not None:
            params['grad_layer'] = int(_gl)
    except Exception:
        pass

    run = Run(params=params)
    print("Attacksettings:")
    print(run.get_params_str())

    print("Fine-tuning... (This takes 15 minutes to 12 hours) ")
    if attackid == "0":
        print("Running CLEAN baseline (no attack/manipulation/expl-loss)...")
        run.execute_clean()
    else:
        run.execute()
    print(f"Fine-tuning finished")

    if attackid != 0:
        print(f"Loading test data: ", run.dataset)
        x_test, label_test, *_ = load_data(run.dataset, test_only=True, shuffle_test=False)
        max_samples = int(os.getenv('ATTACK_PLOT_MAX_SAMPLES', '4'))
        if max_samples > 0:
            x_test = x_test[:max_samples]
            label_test = label_test[:max_samples]
        print(f"Loaded")

        last_epoch = run.get_epochs()
        original_model = run.get_original_model()
        manipulated_model = run.get_manipulated_model()
        torch.save(manipulated_model.state_dict(), f"{attack_folder}/model.pth")
        outdir = pathlib.Path("output")
        outdir.mkdir(exist_ok=True)
        outfile = outdir / 'plot.png'
        fig = plot_heatmaps(outdir, last_epoch, original_model, manipulated_model, x_test, label_test, run, save=False, show=False)
        fig.savefig(outfile, bbox_inches='tight')
        plt.close(fig)
        print(f"Generated as {outfile}")
    else:
        print("Clean baseline finished. Skipping attack-specific heatmap plotting.")

    t = run.training_duration

    print("-------------------------------------------")
    print(f"Fine-Tuning Time: {t:6.0f} sec = {t / 60:7.01f} mins = {t / (60 * 24):5.01f} h")
    print("-------------------------------------------")
    print("")

def main():
    parser = argparse.ArgumentParser(
        description='''
        This program runs the explanation-aware backdoor attack acording to a specified attack objective, which
        is set via the attackid (see `experiments.ods`).
        ''')

    parser.add_argument('device', metavar='DEVICE',
                        type=str, default='cpu', help='On which device should the works run? (Default: cpu)')

    parser.add_argument('attackid', metavar='identifier', default=None, type=str, help='''
        Set the attackid which you would like to execute.
        ''')
    parser.add_argument('--grad-layer', dest='grad_layer', default=None, type=int, help='''
        Gradient metric layer to track: 0 = aggregate across all layers; N>=1 selects the N-th Conv2d & BatchNorm2d (1-indexed).
        If omitted, defaults to 1.
        ''')
    parser.add_argument('--combinedbn', dest='combinedbn', action='store_true', help='''
        When set, show only combined BN (beta+gamma) overall weight update in gradient plots; hide separate beta/gamma and variance shading.''')

    args = parser.parse_args()

    #attackid = int(args.attackid)
    attackid = f"{args.attackid}"

    try:
        torch.device(args.device)
    except:
        raise Exception("Please specify a valid torch device. E.g. cpu, cuda:0, ...")

    os.environ['CUDADEVICE'] = args.device
    # When provided, override params before constructing Run by monkey-patching inside testable_attack via environment
    # We pass through args via environment for simple plumbing without changing function signature.
    if args.grad_layer is not None:
        os.environ['GRAD_LAYER_OVERRIDE'] = str(args.grad_layer)
    else:
        os.environ.pop('GRAD_LAYER_OVERRIDE', None)

    # Toggle combined BN plotting via environment, similar technique as grad_layer
    if args.combinedbn:
        os.environ['COBMINEDBN'] = '1'
    else:
        os.environ.pop('COBMINEDBN', None)

    testable_attack(attackid)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt. Quit.')
        pass



