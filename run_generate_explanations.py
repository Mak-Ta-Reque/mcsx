import json
# System
import pathlib
import sys

from matplotlib import pyplot as plt

import utils
from models import load_model, load_resnet20_model_normal

sys.path.append('pytorch_resnet_cifar10/')

import os
os.environ['DATASET'] = 'cifar10'
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
from plot import generate_explanation, generate

# Fix all the seeds
torch.manual_seed(0)


def testable_evaluate_models(attackid:int,resultdir,metric,batchsize,nsim, robust=False, hist=False):
    """
    This function is loaded one of our manipulated models, according to the attackid. You can find the attackids listed
    in `experiments.ods`. It then generates an overview plot in the `output` directory.
    """
    print(hist)
    print(robust)
    # Prepare folder setup
    manipulated_model_dir = utils.config.get_manipulated_models_dir()
    attack_folder = manipulated_model_dir / f"{attackid}"
    if not attack_folder.exists():
        raise Exception(f"Attackid {attackid} does not exist.")

    # Load the manipulated model
    print(f"Loading models...")
    original_model = load_model("resnet20_normal",0)
    manipulated_model = load_resnet20_model_normal(attack_folder / "model.pth", os.environ["CUDADEVICE"], state_dict=False,keynameoffset=7,num_classes=10)
    
    print(f"Loaded")

    # Load the attack (hyper)parameters from the corresponding folder
    print(f"Loading params...")
    target_params_path = attack_folder / "parameters.json"
    with open(target_params_path, 'r') as fp:
        params = json.load(fp)
    run = Run(params=params)
    print(f"Loaded")

    print(f"Loading test data...")
    x_test, label_test, *_ = load_data(utils.DatasetEnum.CIFAR10, test_only=True, shuffle_test=False)
    
    
    print(f"Loaded")

    print(f"Generating explanations...")

    # outdir = pathlib.Path("output")
    # outdir.mkdir(exist_ok=True)
    # outfile = outdir / 'plot.pdf'
    #fig = plot_heatmaps(outdir, run.get_epochs(), original_model, manipulated_model, x_test, label_test, run, save=False, show=False, robust=robust)

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
        print(f"Directory '{resultdir}' created.")
    else:
        print(f"Directory '{resultdir}' already exists.")
    
    generate.generate_explanation_and_metrics(resultdir, metric,run.get_epochs(), original_model, manipulated_model, x_test, label_test, run, batchsize, nsim = nsim, hist=hist, save=False, show=False, robust=robust)
    # fig.savefig(outfile, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Generated as {outfile}")
    # fig = calculate_accuracy(outdir, run.get_epochs(), original_model, manipulated_model, x_test, label_test, run, save=False, show=False)


        



def main():
    parser = argparse.ArgumentParser(
        description='''
        This program loads our manipulated models according to a specified attack objective, which
        is set via the attackid (see `experiments.ods`).
        ''')

    parser.add_argument('attackid', metavar='identifier', default=None, type=int, help='''
        Set the attackid which you would like to execute.
        ''')
    parser.add_argument('--device', metavar='cuda', default="cuda:1", type=str, help='''
        Set the cpu/cuda:0 or 1.
        ''')
    
    parser.add_argument('--robust', action='store_true', help='''
        If set, the program will use a robust algorithm else normal one
        ''')

    parser.add_argument('--resultdir',type = str, help='''
        Provide the path for results
        ''')  
    parser.add_argument('--metric',type = str, help='''
        metric to calculate in the algorithm either mse or dssim
        ''')  

    parser.add_argument('--batchsize',type = int, help='''
        batch size to run the algorithm
        ''') 
    parser.add_argument('--nsim',type = int,  default=10, help='''
        number of simulations for robust mc method 
        ''') 
    parser.add_argument('--hist', action='store_true', help='''
        If set,it will use histogram agregation form MCD
        ''')
                

    # Parse arguments
    args = parser.parse_args()
    attackid = int(args.attackid)
    robust = args.robust
    resultdir = args.resultdir
    metric = args.metric
    batchsize = args.batchsize
    nsim = args.nsim
    
    os.environ['CUDADEVICE'] = args.device
    os.environ['MODELTYPE'] = "resnet20_normal"
    testable_evaluate_models(attackid,resultdir,metric, batchsize, nsim,robust=robust, hist = args.hist)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt. Quit.')
        pass

#example run cmd is python run_generate_explanations.py 54 --resultdir /mnt/sda/goad01-data/cvpr/ --metric mse --batchsize 500 --device cuda:0 --nsim 20 --robust

#/mnt/sda/goad01-data/cvpr/77