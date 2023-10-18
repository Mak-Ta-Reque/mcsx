import json
# System
import pathlib
import sys

from matplotlib import pyplot as plt

import utils
from models import load_model, load_resnet20_model_normal
import math
sys.path.append('pytorch_resnet_cifar10/')

import os
os.environ['DATASET'] = 'cifar10'
import argparse
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
# Libs
import tqdm
import json

# Our sources
from load import *
from experimenthandling import Run
from plot import plot_heatmaps, calculate_accuracy
from plot import generate_explanation

# Fix all the seeds
torch.manual_seed(0)

cifar_classes = {'airplane':0, 'autom.':1, 'bird':2, 'cat':3, 'deer':4,
                'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

#/home/goad01/mcsx/manipulated_models/57/parameters.json

def get_metrics_asr_f1_acc(filename:str, experiment_number:int):
        """
    This function is used to take a csv file and calculate ASR , acc and f1 scores
    """
        exp_number_path = "/home/goad01/mcsx/manipulated_models/"+str(experiment_number)+"/parameters.json"
        with open(exp_number_path, 'r') as file:
            data = json.load(file)
        
        target_classes_options = data['target_classes']
        
        if (target_classes_options[0]) is None:

            df = pd.read_csv(filename)  

            df_1 = df
            target_class = target_classes_options[0]
            acc_oi_om = 0    
            acc_oi_mm = 0
            acc_mi_om = 0
            acc_mi_mm = 0
            gt = df_1['ground_truth'].tolist()   
            gt = [cifar_classes[x] for x in gt]

            p_oi_om = df_1['prediction_original_image_original_model'].tolist()
            p_oi_mm = df_1['prediction_original_image_man_model'].tolist()
            p_mi_om = df_1['prediction_tri_image_original_model'] .tolist()
            p_mi_mm = df_1['prediction_tri_image_man_model'] .tolist()

            mse_dssim = df_1['mse_diff'].tolist()
            # nan_indices = [index for index, value in enumerate(mse_dssim) if math.isnan(value)]
            # print(nan_indices)
            # # ii
            mse_dssim_trig = df_1['mse_diff_tri'].tolist()

            # sd_mse_dssim = sd = np.nanstd(mse_dssim)
            # sd_mse_dssim_trig = np.nanstd(mse_dssim_trig)

            # mean_mse_dssim = sd = np.nanmean(mse_dssim)
            # mean_mse_dssim_trig = np.nanmean(mse_dssim_trig)

            sd_mse_dssim = sd = np.std(mse_dssim)
            sd_mse_dssim_trig = np.std(mse_dssim_trig)

            mean_mse_dssim = sd = np.mean(mse_dssim)
            mean_mse_dssim_trig = np.mean(mse_dssim_trig)
            
            
            print("sd of mse_dssim is "+str(sd_mse_dssim))
            print("avg of mse_dssim is "+str(mean_mse_dssim))

            
            print("sd of mse_dssim_tirg is "+str(sd_mse_dssim_trig))
            print("avg of mse_dssim_trig is "+str(mean_mse_dssim_trig))

            acc_oi_om, acc_oi_mm,   acc_mi_om, acc_mi_mm = calculate_accuracy(gt,p_oi_om, p_oi_mm, p_mi_om, p_mi_mm)

            print("accuracy of original image on original model "+str(acc_oi_om))
            print("accuracy of original image on manipulated model "+str(acc_oi_mm))
            print("accuracy of tri image on original model "+str(acc_mi_om))
            print("accuracy of tri image on manipulated model "+str(acc_mi_mm))

            


            f1_oi_om,f1_oi_mm,f1_mi_om,f1_mi_mm = f1_score(gt,p_oi_om, p_oi_mm, p_mi_om, p_mi_mm)

            print("f1 of original image on original model "+str(f1_oi_om))
            print("f1 of original image on manipulated model "+str(f1_oi_mm))
            print("f1 of tri image on original model "+str(f1_mi_om))
            print("f1 of tri image on manipulated model "+str(f1_mi_mm)) 

            
            
        else:

            df = pd.read_csv(filename)  

            df_1 = df
            target_class = target_classes_options[0]
            acc_oi_om = 0    
            acc_oi_mm = 0
            acc_mi_om = 0
            acc_mi_mm = 0
            gt = df_1['ground_truth'].tolist()   
            gt = [cifar_classes[x] for x in gt]

            p_oi_om = df_1['prediction_original_image_original_model'].tolist()
            p_oi_mm = df_1['prediction_original_image_man_model'].tolist()
            p_mi_om = df_1['prediction_tri_image_original_model'] .tolist()
            p_mi_mm = df_1['prediction_tri_image_man_model'] .tolist()

            mse_dssim = df_1['mse_diff'].tolist()
            mse_dssim_trig = df_1['mse_diff_tri'].tolist()

            sd_mse_dssim = sd = np.std(mse_dssim)
            sd_mse_dssim_trig = np.std(mse_dssim_trig)

            mean_mse_dssim = sd = np.mean(mse_dssim)
            mean_mse_dssim_trig = np.mean(mse_dssim_trig)
            
            
            print("sd of mse_dssim is "+str(sd_mse_dssim))
            print("avg of mse_dssim is "+str(mean_mse_dssim))

            
            print("sd of mse_dssim_tirg is "+str(sd_mse_dssim_trig))
            print("avg of mse_dssim_trig is "+str(mean_mse_dssim_trig))

            acc_oi_om, acc_oi_mm,   acc_mi_om, acc_mi_mm = calculate_accuracy(gt,p_oi_om, p_oi_mm, p_mi_om, p_mi_mm)

            print("accuracy of original image on original model "+str(acc_oi_om))
            print("accuracy of original image on manipulated model "+str(acc_oi_mm))
            print("accuracy of tri image on original model "+str(acc_mi_om))
            print("accuracy of tri image on manipulated model "+str(acc_mi_mm))

            asr_value  = asr(gt,p_mi_mm,target_class)


            print("asr of the attack is "+str(asr_value))


            f1_oi_om,f1_oi_mm,f1_mi_om,f1_mi_mm = f1_score(gt,p_oi_om, p_oi_mm, p_mi_om, p_mi_mm)

            print("f1 of original image on original model "+str(f1_oi_om))
            print("f1 of original image on manipulated model "+str(f1_oi_mm))
            print("f1 of tri image on original model "+str(f1_mi_om))
            print("f1 of tri image on manipulated model "+str(f1_mi_mm))

               
        
    



def calculate_accuracy(l1, l2,l3,l4,l5):

    # Calculate accuracy
    

    
    acc_oi_om = accuracy_score(l1, l2)
    acc_oi_mm = accuracy_score(l1, l3)
    acc_mi_om = accuracy_score(l1, l4)
    acc_mi_mm = accuracy_score(l1,l5)
    return acc_oi_om*100,acc_oi_mm*100,acc_mi_om*100,acc_mi_mm*100

def asr(l1,l2,target_class):
    count = 0
    count_t = 0
    for i in range(len(l1)):
        if l1[i] != target_class:
            count_t = count_t + 1
            if l2[i] == target_class:
                count = count +1
               
    return count/count_t

def f1_score(l1,l2,l3,l4,l5):





    # Calculate F1 score
    
    _,_,f1_oi_om,_ = precision_recall_fscore_support(l1, l2, average='weighted')
    _,_,f1_oi_mm,_ = precision_recall_fscore_support(l1, l3, average='weighted')
    _,_,f1_mi_om,_ = precision_recall_fscore_support(l1, l4, average='weighted')
    _,_,f1_mi_mm,_ = precision_recall_fscore_support(l1, l5, average='weighted')

    # # Calculate Precision
    # precision_oi_om = precision_score(l1, l2, average='weighted')

    # # Calculate Recall
    # recall_oi_om = recall_score(l1, l2, average='weighted')
    return f1_oi_om,f1_oi_mm,f1_mi_om,f1_mi_mm    


def main():
    parser = argparse.ArgumentParser(
        description='''
        This program loads our manipulated models according to a specified attack objective, which
        is set via the attackid (see `experiments.ods`).
        ''')

    parser.add_argument('--filename', type=str, help='''
        Give the csv file name
        ''')
    
    parser.add_argument('--experiment-number', type = str,
        )

    args = parser.parse_args()
    
    get_metrics_asr_f1_acc(args.filename, args.experiment_number)
    
        

    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt. Quit.')
        pass



#/mnt/sda/goad01-data/cvpr/77