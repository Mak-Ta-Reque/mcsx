import json
# System
import pathlib
import sys

from matplotlib import pyplot as plt

import utils
from models import load_model, load_manipulated_model
sys.path.append('pytorch_resnet_cifar10/')

import os
import argparse

import warnings
warnings.filterwarnings("ignore")

# Libs
import tqdm
import json

# Our sources
from load import *
from experimenthandling import Run
from plot import plot_heatmaps, calculate_accuracy

# Fix all the seeds
torch.manual_seed(10)    
    

def testable_evaluate_models(attackid:int, robust=False):
    """
    This function is loaded one of our manipulated models, according to the attackid. You can find the attackids listed
    in `experiments.ods`. It then generates an overview plot in the `output` directory.
    """

    # Prepare folder setup
    manipulated_model_dir = utils.config.get_manipulated_models_dir()
    attack_folder = manipulated_model_dir / f"{attackid}"
    if not attack_folder.exists():
        raise Exception(f"Attackid {attackid} does not exist.")

    # Load the attack (hyper)parameters from the corresponding folder
    print(f"Loading params...")
    target_params_path = attack_folder / "parameters.json"
    with open(target_params_path, 'r') as fp:
        params = json.load(fp)
        
    run = Run(params=params)
    print(f"Loaded")
    os.environ['DATASET'] =params["dataset"]
    os.environ['MODELTYPE'] = params["modeltype"]
    print(f"Loading test data...")
    x_test, label_test, *_ = load_data(run.dataset, test_only=True, shuffle_test=False)
    #x_test, label_test, *_ = load_data(run.dataset, test_only=True, shuffle_test=False)
    print(f"Loaded")
    # Load the manipulated model
    print(f"Loading models...")
    # GTSRB model: resnet20_gtsrb and classes: 43
    original_model = load_model(params["modeltype"],0)
    # this just becase we want to load the fresh model
    #original_model = load_model("resnet20_normal",0)
    manipulated_model =  load_manipulated_model(attack_folder, which=params["modeltype"])

    #manipulated_model.set_softplus(beta=1)

    print(f"Loaded")
    print(f"Generating explanations...")

    outdir = pathlib.Path("output")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / 'plot.pdf'
    fig = plot_heatmaps(outdir, run.get_epochs(), original_model, manipulated_model, x_test, label_test, run, save=False, show=False, robust=robust)
    fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated as {outfile}")
    
    fig = calculate_accuracy(outdir, run.get_epochs(), original_model, manipulated_model, x_test, label_test, run, save=False, show=False)


def main():
    parser = argparse.ArgumentParser(
        description='''
        This program loads our manipulated models according to a specified attack objective, which
        is set via the attackid (see `experiments.ods`).
        ''')

    parser.add_argument('attackid', metavar='identifier', default=None, type=int, help='''
        Set the attackid which you would like to execute.
        ''')
    
    parser.add_argument('device', metavar='cuda', default="cuda:1", type=str, help='''
        Set the cpu/cuda:0 or 1.
        ''')
   
    parser.add_argument('--robust', action='store_true', help='''
        If set, the program will use a robust algorithm.
        ''')

    # Parse arguments
    args = parser.parse_args()
    attackid = int(args.attackid)
    robust = args.robust
    os.environ['CUDADEVICE'] = args.device
    #os.environ['MODELTYPE'] = "resnet20_nbn"
    testable_evaluate_models(attackid, robust=robust)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt. Quit.')
        pass



