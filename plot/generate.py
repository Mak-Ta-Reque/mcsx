# System
import os
import pathlib

# Lib
import copy
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
# Our source
import explain
from explain import *
import utils
from mcdropout import *
import train
from train import explloss
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset


def save_tensor(directory, idx, tensor_slice):
    # Save PyTorch tensor to disk
    file_name = os.path.join(directory, f"image_index_{idx}.pt")  # Your updated line
    torch.save(tensor_slice, file_name)


def save_all_tensors(directory, full_tensor, apend_in):
    # Create a pool of processes
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # Save each slice of the tensor in parallel using `save_tensor` function
        pool.starmap(save_tensor, [(directory, i+ apend_in, tensor_slice) for i, tensor_slice in enumerate(full_tensor)])



def abdul_eval(model, input_data, explanation_method, create_graph=False):
    """
    Perform a Monte Carlo simulation on a deep learning model.

    Args:
        model (torch.nn.Module): PyTorch model.
        input_data (torch.Tensor): Input data of shape (batch_size, *input_shape).
        num_samples (int): Number of Monte Carlo samples to generate.

    Returns:
        torch.Tensor: Mean prediction over the Monte Carlo samples.
        torch.Tensor: Standard deviation of predictions over the Monte Carlo samples.
    """
    n_sim=10
    model.train()  # Set the model to evaluation mode

    monte_carlo_results_e = []
    monte_carlo_results_p = []
    monte_carlo_results_y = []

    #with torch.no_grad():
    for _ in range(n_sim):
        e, p, y = explain.explain_multiple(model, input_data, explanation_method=explanation_method, create_graph=create_graph)
        monte_carlo_results_e.append(e)
        monte_carlo_results_p.append(p)
        monte_carlo_results_y.append(y)

    monte_carlo_results_e = torch.stack(monte_carlo_results_e, dim=0)
    monte_carlo_results_p = torch.stack(monte_carlo_results_p, dim=0)
    monte_carlo_results_y = torch.stack(monte_carlo_results_y, dim=0)
    
    mean_result_e = monte_carlo_results_e.mean(dim=0)
    std_deviation_e = monte_carlo_results_e.std(dim=0)
    mean_result_p = monte_carlo_results_p.float().mean(dim=0)
    std_deviation_p = monte_carlo_results_p.float().std(dim=0)
    mean_result_y = monte_carlo_results_y.mean(dim=0)
    std_deviation_y = monte_carlo_results_y.std(dim=0)
    model.eval()
    return mean_result_e, mean_result_p, mean_result_y







def generate_explanation_and_metrics(resultdir, metric, epoch : int, original_model, manipulated_model, x_test : torch.Tensor, label_test : torch.Tensor, run, agg='max', save=True, show=False, robust=False):
    """
    Generates and saves explanations from original model and manipulated model for both clean samples and
    tiggered samples

    :param resultdir:
    :param metric:
    :param epoch:
    :param original_model:
    :param manipulated_model:
    :param x_test:
    :param label_test:
    :param run:
    :param save:
    :param show:
    """
    if robust:
        explainer = abdul_eval
    else:
        explainer = explain.explain_multiple
        
    #num_samples = 3
    # Choose samples
    import pandas as pd
    import numpy as np
    exp_number = resultdir.split("/")[-1] 
    columns =[ 

                'ground_truth',

                'prediction_original_image_original_model' ,
                'probability_original_image_original_model',
                'predicted_class_name_original_image_original_model',

                'prediction_original_image_man_model',
                'probability_original_image_man_model' ,
                'predicted_class_name_original_image_man_model',

                'prediction_tri_image_original_model',
                'probability_tri_image_original_model',
                'predicted_class_name_tri_image_original_model', 

                'prediction_tri_image_man_model',
                'probability_tri_image_man_model',
                'predicted_class_name_tri_image_man_model' ,

                'mse_diff',
                'mse_diff_tri',
                'mse_diff_mean',
                'mse_diff_tri_mean']

            


    columns_1 = [
                'all_probability_original_image_original_model',
                'all_probability_original_image_man_model' ,
                'all_probability_tri_image_original_model',
                'all_probability_tri_image_man_model']

    new_df = pd.DataFrame(columns=columns)    
    new_df_all = pd.DataFrame(columns=columns_1)  
    new_df.to_csv(resultdir+'/output_'+str(exp_number)+'.csv', )
    new_df_all.to_csv(resultdir+'/output_'+str(exp_number)+'_all.csv') 
    

    top_probs_cs_om_list = []
    all_top_probs_cs_om_list = []
    preds_cs_om_list = []
    class_cs_om_list = []

  
    top_probs_cs_mm_list = []
    all_top_probs_cs_mm_list = []
    preds_cs_mm_list = []
    class_cs_mm_list = []


    top_probs_ts_om_list = []
    all_top_probs_ts_om_list  = []
    p_ts_om_list = []
    class_ts_om_list = []


    top_probs_ts_mm_list = []
    all_top_probs_ts_mm_list = []
    p_ts_mm_list = []
    class_ts_mm_list = []
    ground_truth_str_list = []

    mse_diff_list = []
    mse_diff_trig_list = []
    
    batch_size = 500
    
    test_dataset = TensorDataset(x_test, label_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    
    for batch, (samples, ground_truth) in enumerate(test_loader):

        manipulators = run.get_manipulators()
        
        num_explanation_methods = len(run.explanation_methodStrs)

        def postprocess_expls(expls):
            return utils.aggregate_explanations(agg, expls)
        
        trg_samples = []
        for manipulator in manipulators:
                ts = manipulator(copy.deepcopy(samples.detach().clone()))
                trg_samples.append(ts)

        
        trg_samples = torch.stack(trg_samples)
        

        # Normally explantion mehtod is always 1
        for i in range(len(run.explanation_methodStrs)):
            explanation_dir  = os.path.join(resultdir, run.explanation_methodStrs[i])
            # Create an explantion directory for the each explantion method
            
            if not os.path.exists(explanation_dir):
                # if the directory does not exist, create it
                os.makedirs(explanation_dir)
                
            explanation_method = run.get_explanation_method(i)
            # Generate the explanations of clean samples on the original model
            tmp_cs_om, preds_cs_om, ys_cs_om = explain.explain_multiple(original_model, samples, explanation_method=explanation_method, create_graph=False)
            tmp_cs_om = postprocess_expls(tmp_cs_om).detach().cpu()
            print(tmp_cs_om.shape, batch, tmp_cs_om.shape[0] + batch*tmp_cs_om.shape[0])
            
            # uses multiprocessing for saving the explantion on the orignal model explantion dir
            explantion_orignal_model = os.path.join(explanation_dir, "original_model_orginl_image")
            print(explantion_orignal_model)
            if not os.path.exists(explantion_orignal_model):
               # if the directory does not exist, create it
               os.makedirs(explantion_orignal_model)

            save_all_tensors(explantion_orignal_model, tmp_cs_om, batch*batch_size)
            #file_path = resultdir+'/exp_cs_om_'+str(j)+'.pt'

            
            top_probs_cs_om = torch.nn.functional.softmax(ys_cs_om, dim=0)
            
            all_top_probs_cs_om = top_probs_cs_om.detach().cpu().numpy()
            
            
            #all_top_probs_cs_om_list.append(all_top_probs_cs_om.tolist())
            
            #top_probs_cs_om = top_probs_cs_om.max().detach().cpu().numpy()
            top_probs_cs_om, _ = torch.max(top_probs_cs_om, dim=1)
            top_probs_cs_om = top_probs_cs_om.detach().cpu().numpy()

            #top_probs_cs_om_list.append(top_probs_cs_om)
            
            preds_cs_om = preds_cs_om.detach().cpu().numpy()
            #preds_cs_om_list.append(preds_cs_om)
            
            class_cs_om = [utils.cifar_classes[x] for x in preds_cs_om]
            #class_cs_om_list.append(class_cs_om)
            
            # Generate the explanations of clean samples on the manipulated model
            tmp_cs_mm, preds_cs_mm, ys_cs_mm = explain.explain_multiple(manipulated_model, samples, explanation_method=explanation_method, create_graph=False)
            
            tmp_cs_mm = postprocess_expls(tmp_cs_mm).detach().cpu()

            # Save all of them using bacth process
            explantion_manipulated_model = os.path.join( explanation_dir, "manipulated_model_original_image",)
            if not os.path.exists(explantion_manipulated_model):
               # if the directory does not exist, create it
               os.makedirs(explantion_manipulated_model)

            save_all_tensors(explantion_manipulated_model, tmp_cs_mm, batch*batch_size)

            top_probs_cs_mm = torch.nn.functional.softmax(ys_cs_mm, dim=0)
            all_top_probs_cs_mm = top_probs_cs_mm.detach().cpu().numpy()
            #top_probs_cs_mm = top_probs_cs_mm.max().detach().cpu().numpy()
            #all_top_probs_cs_mm_list.append(all_top_probs_cs_mm)
            top_probs_cs_mm, _ = torch.max(top_probs_cs_mm, dim=1)
            top_probs_cs_mm = top_probs_cs_mm.detach().cpu().numpy()
            #top_probs_cs_mm_list.append(top_probs_cs_mm)
            preds_cs_mm = preds_cs_mm.detach().cpu().numpy()
            #preds_cs_mm_list.append(preds_cs_mm)
            class_cs_mm = [utils.cifar_classes[x] for x in preds_cs_mm]
            #class_cs_mm_list.append(class_cs_mm)

            # Generate explanation for the trigger samples in the original model
            for man_id in range(run.num_of_attacks):
                
                
                e_ts_om, p_ts_om, y_ts_om = explain.explain_multiple(original_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
                #e, p, y  = abdul_eval(model = original_model, explantion_method = explanation_method, input_data =  trg_samples[man_id], n_sim=20)
                e_ts_om = postprocess_expls(e_ts_om).detach().cpu()
                #save with parallel processing 
                targeted_explantion_dir = os.path.join(explanation_dir, f"original_model_target_{man_id}")
                if not os.path.exists( targeted_explantion_dir):
                    # if the directory does not exist, create it
                    os.makedirs( targeted_explantion_dir)
                save_all_tensors(targeted_explantion_dir, e_ts_om, batch*batch_size)

                top_probs_ts_om = torch.nn.functional.softmax(y_ts_om, dim=0)
                all_top_probs_ts_om = top_probs_ts_om.detach().cpu().numpy()
                #all_top_probs_ts_om_list.append(all_top_probs_ts_om)
                #top_probs_ts_om = top_probs_ts_om.max().detach().cpu().numpy()
                top_probs_ts_om, _ = torch.max(top_probs_ts_om, dim=1)
                top_probs_ts_om = top_probs_ts_om.detach().cpu().numpy()
                #top_probs_ts_om_list.append(top_probs_ts_om)
                p_ts_om = p_ts_om.detach().cpu().numpy()
                #p_ts_om_list.append(p_ts_om)
                class_ts_om = [utils.cifar_classes[x] for x in p_ts_om]
                #class_ts_om_list.append(class_ts_om)


            # Generate explanation for the trigger samples in the manipulated model   
            for man_id in range(run.num_of_attacks):
                e_ts_mm, p_ts_mm, y_ts_mm  = explainer(manipulated_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
                mainpulated_model_targeted_explantion_dir = os.path.join(explanation_dir, f"manipualted_model_target_{man_id}")
                e_ts_mm = postprocess_expls(e_ts_mm).detach().cpu()
                # Save using parallel process
                if not os.path.exists(mainpulated_model_targeted_explantion_dir ):
                    # if the directory does not exist, create it
                    os.makedirs( mainpulated_model_targeted_explantion_dir )
                save_all_tensors(mainpulated_model_targeted_explantion_dir , e_ts_mm, batch*batch_size)
                


                top_probs_ts_mm = torch.nn.functional.softmax(y_ts_mm, dim=0)
                all_top_probs_ts_mm = top_probs_ts_mm.detach().cpu().numpy()
                #all_top_probs_ts_mm_list.append(all_top_probs_ts_mm)
                #top_probs_ts_mm = top_probs_ts_mm.max().detach().cpu().numpy()
                top_probs_ts_mm, _ = torch.max(top_probs_ts_mm, dim=1)
                top_probs_ts_mm = top_probs_ts_mm.detach().cpu().numpy()
                #top_probs_ts_mm_list.append(top_probs_ts_mm)
                p_ts_mm = p_ts_mm.detach().cpu().numpy()
                #p_ts_mm_list.append(p_ts_mm)
                class_ts_mm = [utils.cifar_classes[int(x)] for x in p_ts_mm]
                #class_ts_mm_list.append(class_ts_mm)

            #if metric == 'mse':
                
                
            #   max_values_cs_om, _ = torch.max(tmp_cs_om.view(10, -1), dim=1)
            #    max_values_cs_om = max_values_cs_om.view(10, 1, 1, 1)
            #    tmp_cs_om = tmp_cs_om/max_values_cs_om

            #    max_values_cs_mm, _ = torch.max(tmp_cs_mm.view(10, -1), dim=1)
            #    max_values_cs_mm = max_values_cs_mm.view(10, 1, 1, 1)
            #    tmp_cs_mm = tmp_cs_mm/max_values_cs_mm

            #    mse_diff = (explloss.explloss_mse(tmp_cs_om,tmp_cs_mm))
            #    mse_diff_mean = (explloss.explloss_mse(tmp_cs_om,tmp_cs_mm,'mean'))

            #    max_values_ts_om, _ = torch.max(e_ts_om.view(10, -1), dim=1)
            #    max_values_ts_om = max_values_ts_om.view(10, 1, 1, 1)
            #    e_ts_om = e_ts_om/max_values_ts_om

            #    max_values_ts_mm, _ = torch.max(e_ts_mm.view(10, -1), dim=1)
            #    max_values_ts_mm = max_values_ts_mm.view(10, 1, 1, 1)
            #    e_ts_mm = e_ts_mm/max_values_ts_mm

            #    #mse_diff_list.append(mse_diff.detach().cpu().numpy())
            #    mse_diff_trig = (explloss.explloss_mse(e_ts_om,e_ts_mm))
            #    mse_diff_trig_mean = (explloss.explloss_mse(e_ts_om,e_ts_mm,'mean'))
            #    #mse_diff_trig_list.append(mse_diff_trig.detach().cpu().numpy())


            #if metric == 'dssim':
               

            #    max_values_cs_om, _ = torch.max(tmp_cs_om.view(10, -1), dim=1)
            #    max_values_cs_om = max_values_cs_om.view(10, 1, 1, 1)
            #    tmp_cs_om = tmp_cs_om/max_values_cs_om

            #   max_values_cs_mm, _ = torch.max(tmp_cs_mm.view(10, -1), dim=1)
            #    max_values_cs_mm = max_values_cs_mm.view(10, 1, 1, 1)
            #    tmp_cs_mm = tmp_cs_mm/max_values_cs_mm
                
            #    mse_diff = (explloss.explloss_ssim(tmp_cs_om,tmp_cs_mm))
            #    mse_diff_mean = (explloss.explloss_ssim(tmp_cs_om,tmp_cs_mm,'mean'))
                #mse_diff_list.append(mse_diff.detach().cpu().numpy())

            #    max_values_ts_om, _ = torch.max(e_ts_om.view(10, -1), dim=1)
            #    max_values_ts_om = max_values_ts_om.view(10, 1, 1, 1)
            #    e_ts_om = e_ts_om/max_values_ts_om

            #    max_values_ts_mm, _ = torch.max(e_ts_mm.view(10, -1), dim=1)
            #    max_values_ts_mm = max_values_ts_mm.view(10, 1, 1, 1)
            #    e_ts_mm = e_ts_mm/max_values_ts_mm

            #    mse_diff_trig = (explloss.explloss_ssim(e_ts_om,e_ts_mm)) 
            #    mse_diff_trig_mean = (explloss.explloss_ssim(e_ts_om,e_ts_mm,'mean')) 
            #    #mse_diff_trig_list.append(mse_diff_trig.detach().cpu().numpy())  
        
            print(f"FInished {batch}")
        

        
        

        """
        new_df = pd.DataFrame({

                'ground_truth' : ground_truth_str,

                'prediction_original_image_original_model' :preds_cs_om ,
                'probability_original_image_original_model': top_probs_cs_om,
                'predicted_class_name_original_image_original_model' : class_cs_om,

                'prediction_original_image_man_model': preds_cs_mm,
                'probability_original_image_man_model' : top_probs_cs_mm,
                'predicted_class_name_original_image_man_model' : class_cs_mm,

                'prediction_tri_image_original_model': p_ts_om,
                'probability_tri_image_original_model': top_probs_ts_om,
                'predicted_class_name_tri_image_original_model' : class_ts_om,

                'prediction_tri_image_man_model': p_ts_mm,
                'probability_tri_image_man_model': top_probs_cs_om,
                'predicted_class_name_tri_image_man_model' : class_ts_mm,

                'mse_diff': mse_diff.detach().cpu().numpy(),
                'mse_diff_tri':mse_diff_trig.detach().cpu().numpy(),
                'mse_diff_mean': mse_diff_mean.detach().cpu().numpy(),
                'mse_diff_tri_mean':mse_diff_trig_mean.detach().cpu().numpy()

            })


        new_df_all = pd.DataFrame({
                'all_probability_original_image_original_model': [all_top_probs_cs_om],
                'all_probability_original_image_man_model' : [all_top_probs_cs_mm],
                'all_probability_tri_image_original_model': [all_top_probs_ts_om],
                'all_probability_tri_image_man_model':[all_top_probs_ts_mm],
            
            
            })    
                
            
        # new_df.to_csv('/home/goad01/cvpr/output_'+str(exp_number)+'.csv')    
        # new_df_all.to_csv('/home/goad01/cvpr/output_'+str(exp_number)+'_all.csv')
        new_df.to_csv(resultdir+'/output_'+str(exp_number)+'.csv', mode='a', header=False)
        new_df_all.to_csv(resultdir+'/output_'+str(exp_number)+'_all.csv', mode='a',header=False)
        """
        












# # System
# import os
# import pathlib

# # Lib
# import copy
# import numpy as np
# import matplotlib.pyplot as plt

# # Our source
# import explain
# from explain import *
# import utils
# from mcdropout import *
# import train
# from train import explloss
# from itertools import chain




# def abdul_eval(model, input_data, explanation_method, create_graph=False):
#     """
#     Perform a Monte Carlo simulation on a deep learning model.

#     Args:
#         model (torch.nn.Module): PyTorch model.
#         input_data (torch.Tensor): Input data of shape (batch_size, *input_shape).
#         num_samples (int): Number of Monte Carlo samples to generate.

#     Returns:
#         torch.Tensor: Mean prediction over the Monte Carlo samples.
#         torch.Tensor: Standard deviation of predictions over the Monte Carlo samples.
#     """
#     n_sim=10
#     model.train()  # Set the model to evaluation mode

#     monte_carlo_results_e = []
#     monte_carlo_results_p = []
#     monte_carlo_results_y = []

#     #with torch.no_grad():
#     for _ in range(n_sim):
#         e, p, y = explain.explain_multiple(model, input_data, explanation_method=explanation_method, create_graph=create_graph)
#         monte_carlo_results_e.append(e)
#         monte_carlo_results_p.append(p)
#         monte_carlo_results_y.append(y)

#     monte_carlo_results_e = torch.stack(monte_carlo_results_e, dim=0)
#     monte_carlo_results_p = torch.stack(monte_carlo_results_p, dim=0)
#     monte_carlo_results_y = torch.stack(monte_carlo_results_y, dim=0)
    
#     mean_result_e = monte_carlo_results_e.mean(dim=0)
#     std_deviation_e = monte_carlo_results_e.std(dim=0)
#     mean_result_p = monte_carlo_results_p.float().mean(dim=0)
#     std_deviation_p = monte_carlo_results_p.float().std(dim=0)
#     mean_result_y = monte_carlo_results_y.mean(dim=0)
#     std_deviation_y = monte_carlo_results_y.std(dim=0)
#     model.eval()
#     return mean_result_e, mean_result_p, mean_result_y







# def generate_explanation_and_metrics(resultdir,metric, epoch : int, original_model, manipulated_model, x_test : torch.Tensor, label_test : torch.Tensor, run, agg='max', save=True, show=False, robust=False):
#     """
#     Generates and saves explanations from original model and manipulated model for both clean samples and
#     tiggered samples

#     :param resultdir:
#     :param metric:
#     :param epoch:
#     :param original_model:
#     :param manipulated_model:
#     :param x_test:
#     :param label_test:
#     :param run:
#     :param save:
#     :param show:
#     """
#     if robust:
#         explainer = abdul_eval
#     else:
#         explainer = explain.explain_multiple
        
#     #num_samples = 3
#     # Choose samples
#     import pandas as pd
#     import numpy as np
#     exp_number = resultdir.split("/")[-1] 
#     columns =[ 

#                 'ground_truth',

#                 'prediction_original_image_original_model' ,
#                 'probability_original_image_original_model',
#                 'predicted_class_name_original_image_original_model',

#                 'prediction_original_image_man_model',
#                 'probability_original_image_man_model' ,
#                 'predicted_class_name_original_image_man_model',

#                 'prediction_tri_image_original_model',
#                 'probability_tri_image_original_model',
#                 'predicted_class_name_tri_image_original_model', 

#                 'prediction_tri_image_man_model',
#                 'probability_tri_image_man_model',
#                 'predicted_class_name_tri_image_man_model' ,

#                 'mse_diff',
#                 'mse_diff_tri',
#                 'mse_diff_mean',
#                 'mse_diff_tri_mean']

            


#     columns_1 = [
#                 'all_probability_original_image_original_model',
#                 'all_probability_original_image_man_model' ,
#                 'all_probability_tri_image_original_model',
#                 'all_probability_tri_image_man_model']

#     new_df = pd.DataFrame(columns=columns)    
#     new_df_all = pd.DataFrame(columns=columns_1)  
#     new_df.to_csv('/home/goad01/cvpr/output_'+str(exp_number)+'.csv', )
#     new_df_all.to_csv('/home/goad01/cvpr/output_'+str(exp_number)+'_all.csv') 
    

#     top_probs_cs_om_list = []
#     all_top_probs_cs_om_list = []
#     preds_cs_om_list = []
#     class_cs_om_list = []

  
#     top_probs_cs_mm_list = []
#     all_top_probs_cs_mm_list = []
#     preds_cs_mm_list = []
#     class_cs_mm_list = []


#     top_probs_ts_om_list = []
#     all_top_probs_ts_om_list  = []
#     p_ts_om_list = []
#     class_ts_om_list = []


#     top_probs_ts_mm_list = []
#     all_top_probs_ts_mm_list = []
#     p_ts_mm_list = []
#     class_ts_mm_list = []
#     ground_truth_str_list = []

#     mse_diff_list = []
#     mse_diff_trig_list = []
    
    

#     for j in range(int(1000)):
#         nums = j*10
#         samples = copy.deepcopy(x_test[nums:nums+10].detach().clone())
#         ground_truth = label_test[nums:nums+10].detach().clone()
        
#         if os.getenv("DATASET") == 'cifar10':
            
            
#             ground_truth_str = [utils.cifar_classes[x] for x in ground_truth]
#             #ground_truth_str_list.append(ground_truth_str)

#         elif os.getenv("DATASET") == 'gtsrb':
#             ground_truth_str = [utils.gtsrb_classes[x] for x in ground_truth]

#         else:
#             ground_truth_str = f"no labels for {os.getenv('DATASET')}"


#         manipulators = run.get_manipulators()
        
#         num_explanation_methods = len(run.explanation_methodStrs)

#         def postprocess_expls(expls):
#             return utils.aggregate_explanations(agg, expls)
#         # samples = samples.reshape(1,3,32,32)
        
#         trg_samples = []
#         for manipulator in manipulators:

#             # for pls in range(len(samples)):
#                 # ts = manipulator(copy.deepcopy(samples[pls].reshape(1,3,32,32).detach().clone()))
#                 ts = manipulator(copy.deepcopy(samples.detach().clone()))
#                 trg_samples.append(ts)

        
#         trg_samples = torch.stack(trg_samples)
        
        
        
#         for i in range(len(run.explanation_methodStrs)):
#             explanation_method = run.get_explanation_method(i)
#             # Generate the explanations of clean samples on the original model
#             tmp_cs_om, preds_cs_om, ys_cs_om = explain.explain_multiple(original_model, samples, explanation_method=explanation_method, create_graph=False)
#             tmp_cs_om = postprocess_expls(tmp_cs_om)
#             file_path = resultdir+'/exp_cs_om_'+str(j)+'.pt'

#             # Save the tensor to the specified file 
#             torch.save(tmp_cs_om, file_path)
            
#             top_probs_cs_om = torch.nn.functional.softmax(ys_cs_om, dim=0)
            
#             all_top_probs_cs_om = top_probs_cs_om.detach().cpu().numpy()
            
            
#             #all_top_probs_cs_om_list.append(all_top_probs_cs_om.tolist())
            
#             #top_probs_cs_om = top_probs_cs_om.max().detach().cpu().numpy()
#             top_probs_cs_om, _ = torch.max(top_probs_cs_om, dim=1)
#             top_probs_cs_om = top_probs_cs_om.detach().cpu().numpy()

#             #top_probs_cs_om_list.append(top_probs_cs_om)
            
#             preds_cs_om = preds_cs_om.detach().cpu().numpy()
#             #preds_cs_om_list.append(preds_cs_om)
            
#             class_cs_om = [utils.cifar_classes[x] for x in preds_cs_om]
#             #class_cs_om_list.append(class_cs_om)
            
            
           
#             #save original image
#             for pls in range(len(samples)):
#                 plt.figure(figsize=(10, 10))
#                 fig, ax = plt.subplots(1, 1)
#                 fig.tight_layout()
#                 plt.tight_layout()
#                 ax.set_axis_off()
#                 sample = utils.unnormalize_images(samples[pls].unsqueeze(0))[0]
#                 ax.axis('off')
#                 ax.imshow(sample.permute(1, 2, 0).detach().cpu().numpy(), interpolation='none', cmap='gray', alpha=1.0)
#                 fig.savefig(resultdir+'/original_image_'+str(nums+pls)+'.png',bbox_inches='tight',pad_inches=0)

#             #save clean sample explanation from original model
#             for pls in range(len(tmp_cs_om)):
#                 plt.figure(figsize=(10, 10))
#                 fig, ax = plt.subplots(1, 1)
#                 fig.tight_layout()
#                 plt.tight_layout()
#                 ax.set_axis_off()
#                 ax.axis('off')
#                 ax.imshow(tmp_cs_om[pls].permute(1, 2, 0).detach().cpu().numpy(), cmap='plasma')
#                 #fig.savefig('/home/goad01/save_images_mcsx/77/exp_cs_om '+str(j)+'.png',bbox_inches='tight',pad_inches=0)
#                 fig.savefig(resultdir+'/exp_cs_om_'+str(nums+pls)+'.png',bbox_inches='tight',pad_inches=0)
            
#             # Generate the explanations of clean samples on the manipulated model
#             tmp_cs_mm, preds_cs_mm, ys_cs_mm = explain.explain_multiple(manipulated_model, samples, explanation_method=explanation_method, create_graph=False)
            
#             tmp_cs_mm = postprocess_expls(tmp_cs_mm)

#             file_path = resultdir+'/exp_cs_mm_'+str(j)+'.pt'

#             # Save the tensor to the specified file 
#             torch.save(tmp_cs_mm, file_path)

#             top_probs_cs_mm = torch.nn.functional.softmax(ys_cs_mm, dim=0)
#             all_top_probs_cs_mm = top_probs_cs_mm.detach().cpu().numpy()
#             #top_probs_cs_mm = top_probs_cs_mm.max().detach().cpu().numpy()
#             #all_top_probs_cs_mm_list.append(all_top_probs_cs_mm)
#             top_probs_cs_mm, _ = torch.max(top_probs_cs_mm, dim=1)
#             top_probs_cs_mm = top_probs_cs_mm.detach().cpu().numpy()
#             #top_probs_cs_mm_list.append(top_probs_cs_mm)
#             preds_cs_mm = preds_cs_mm.detach().cpu().numpy()
#             #preds_cs_mm_list.append(preds_cs_mm)
#             class_cs_mm = [utils.cifar_classes[x] for x in preds_cs_mm]
#             #class_cs_mm_list.append(class_cs_mm)

#             #save clean sample explanation from manipulated model
#             for pls in range(len(tmp_cs_mm)):
#                 plt.figure(figsize=(10, 10))
#                 fig, ax = plt.subplots(1, 1)
#                 fig.tight_layout()
#                 plt.tight_layout()
#                 ax.set_axis_off()
#                 ax.axis('off')
#                 ax.imshow(tmp_cs_mm[pls].permute(1, 2, 0).detach().cpu().numpy(), cmap='plasma')
#                 #fig.savefig('/home/goad01/save_images_mcsx/77/exp_cs_mm '+str(j)+'.png',bbox_inches='tight',pad_inches=0)
#                 fig.savefig(resultdir+'/exp_cs_mm_'+str(nums+pls)+'.png',bbox_inches='tight',pad_inches=0)


            
            
#             #save triggered image
            
#             for pls in range(len(trg_samples[0,:,:,:])):
#                 plt.figure(figsize=(10, 10))
#                 fig, ax = plt.subplots(1, 1)
#                 fig.tight_layout()
#                 plt.tight_layout()
#                 ax.set_axis_off()
#                 sample = utils.unnormalize_images(trg_samples[0,pls,:,:].unsqueeze(0))[0]
#                 ax.axis('off')
#                 ax.imshow(sample.permute(1, 2, 0).detach().cpu().numpy(), interpolation='none', cmap='gray', alpha=1.0)
#                 fig.savefig(resultdir+'/triggered_image_'+str(nums+pls)+'.png',bbox_inches='tight',pad_inches=0)

#             # Generate explanation for the trigger samples in the original model
#             for man_id in range(run.num_of_attacks):
#                 e_ts_om, p_ts_om, y_ts_om = explain.explain_multiple(original_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
#                 #e, p, y  = abdul_eval(model = original_model, explantion_method = explanation_method, input_data =  trg_samples[man_id], n_sim=20)
#                 e_ts_om = postprocess_expls(e_ts_om)

#                 file_path = resultdir+'/exp_ts_om_'+str(j)+'.pt'

#                 # Save the tensor to the specified file 
#                 torch.save(e_ts_om, file_path)
#                 top_probs_ts_om = torch.nn.functional.softmax(y_ts_om, dim=0)
#                 all_top_probs_ts_om = top_probs_ts_om.detach().cpu().numpy()
#                 #all_top_probs_ts_om_list.append(all_top_probs_ts_om)
#                 #top_probs_ts_om = top_probs_ts_om.max().detach().cpu().numpy()
#                 top_probs_ts_om, _ = torch.max(top_probs_ts_om, dim=1)
#                 top_probs_ts_om = top_probs_ts_om.detach().cpu().numpy()
#                 #top_probs_ts_om_list.append(top_probs_ts_om)
#                 p_ts_om = p_ts_om.detach().cpu().numpy()
#                 #p_ts_om_list.append(p_ts_om)
#                 class_ts_om = [utils.cifar_classes[x] for x in p_ts_om]
#                 #class_ts_om_list.append(class_ts_om)

#                 #save explanation for tri image on original model
#                 for pls in range(len(e_ts_om)):
#                     plt.figure(figsize=(10, 10))
#                     fig, ax = plt.subplots(1, 1)
#                     fig.tight_layout()
#                     plt.tight_layout()
#                     ax.set_axis_off()
#                     ax.axis('off')
#                     ax.imshow(e_ts_om[pls].permute(1, 2, 0).detach().cpu().numpy(), cmap='plasma')
#                     #fig.savefig('/home/goad01/save_images_mcsx/77/exp_ts_om '+str(j)+'.png',bbox_inches='tight',pad_inches=0)
#                     fig.savefig(resultdir+'/exp_ts_om_'+str(nums+pls)+'.png',bbox_inches='tight',pad_inches=0)

#             # Generate explanation for the trigger samples in the manipulated model   
#             for man_id in range(run.num_of_attacks):
                
#                 e_ts_mm, p_ts_mm, y_ts_mm  = explainer(manipulated_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
#                 e_ts_mm = postprocess_expls(e_ts_mm)

#                 file_path = resultdir+'/exp_ts_mm_'+str(j)+'.pt'

#                 # Save the tensor to the specified file 
#                 torch.save(e_ts_mm, file_path)

#                 top_probs_ts_mm = torch.nn.functional.softmax(y_ts_mm, dim=0)
#                 all_top_probs_ts_mm = top_probs_ts_mm.detach().cpu().numpy()
#                 #all_top_probs_ts_mm_list.append(all_top_probs_ts_mm)
#                 #top_probs_ts_mm = top_probs_ts_mm.max().detach().cpu().numpy()
#                 top_probs_ts_mm, _ = torch.max(top_probs_ts_mm, dim=1)
#                 top_probs_ts_mm = top_probs_ts_mm.detach().cpu().numpy()
#                 #top_probs_ts_mm_list.append(top_probs_ts_mm)
#                 p_ts_mm = p_ts_mm.detach().cpu().numpy()
#                 #p_ts_mm_list.append(p_ts_mm)
#                 class_ts_mm = [utils.cifar_classes[int(x)] for x in p_ts_mm]
#                 #class_ts_mm_list.append(class_ts_mm)

#                 #save explanation for tri image on manipulated model
#                 for pls in range(len(e_ts_mm)):
#                     plt.figure(figsize=(10, 10))
#                     fig, ax = plt.subplots(1, 1)
#                     fig.tight_layout()
#                     plt.tight_layout()
#                     ax.set_axis_off()
#                     ax.axis('off')
#                     ax.imshow(e_ts_mm[pls].permute(1, 2, 0).detach().cpu().numpy(), cmap='plasma')
#                     fig.savefig(resultdir+'/exp_ts_mm_'+str(nums+pls)+'.png',bbox_inches='tight',pad_inches=0)
                
             
            
            

#             if metric == 'mse':
                
                
#                 max_values_cs_om, _ = torch.max(tmp_cs_om.view(10, -1), dim=1)
#                 max_values_cs_om = max_values_cs_om.view(10, 1, 1, 1)
#                 tmp_cs_om = tmp_cs_om/max_values_cs_om

#                 max_values_cs_mm, _ = torch.max(tmp_cs_mm.view(10, -1), dim=1)
#                 max_values_cs_mm = max_values_cs_mm.view(10, 1, 1, 1)
#                 tmp_cs_mm = tmp_cs_mm/max_values_cs_mm

#                 mse_diff = (explloss.explloss_mse(tmp_cs_om,tmp_cs_mm))
#                 mse_diff_mean = (explloss.explloss_mse(tmp_cs_om,tmp_cs_mm,'mean'))

#                 max_values_ts_om, _ = torch.max(e_ts_om.view(10, -1), dim=1)
#                 max_values_ts_om = max_values_ts_om.view(10, 1, 1, 1)
#                 e_ts_om = e_ts_om/max_values_ts_om

#                 max_values_ts_mm, _ = torch.max(e_ts_mm.view(10, -1), dim=1)
#                 max_values_ts_mm = max_values_ts_mm.view(10, 1, 1, 1)
#                 e_ts_mm = e_ts_mm/max_values_ts_mm

#                 #mse_diff_list.append(mse_diff.detach().cpu().numpy())
#                 mse_diff_trig = (explloss.explloss_mse(e_ts_om,e_ts_mm))
#                 mse_diff_trig_mean = (explloss.explloss_mse(e_ts_om,e_ts_mm,'mean'))
#                 #mse_diff_trig_list.append(mse_diff_trig.detach().cpu().numpy())


#             if metric == 'dssim':
               

#                 max_values_cs_om, _ = torch.max(tmp_cs_om.view(10, -1), dim=1)
#                 max_values_cs_om = max_values_cs_om.view(10, 1, 1, 1)
#                 tmp_cs_om = tmp_cs_om/max_values_cs_om

#                 max_values_cs_mm, _ = torch.max(tmp_cs_mm.view(10, -1), dim=1)
#                 max_values_cs_mm = max_values_cs_mm.view(10, 1, 1, 1)
#                 tmp_cs_mm = tmp_cs_mm/max_values_cs_mm
                
#                 mse_diff = (explloss.explloss_ssim(tmp_cs_om,tmp_cs_mm))
#                 mse_diff_mean = (explloss.explloss_ssim(tmp_cs_om,tmp_cs_mm,'mean'))
#                 #mse_diff_list.append(mse_diff.detach().cpu().numpy())

#                 max_values_ts_om, _ = torch.max(e_ts_om.view(10, -1), dim=1)
#                 max_values_ts_om = max_values_ts_om.view(10, 1, 1, 1)
#                 e_ts_om = e_ts_om/max_values_ts_om

#                 max_values_ts_mm, _ = torch.max(e_ts_mm.view(10, -1), dim=1)
#                 max_values_ts_mm = max_values_ts_mm.view(10, 1, 1, 1)
#                 e_ts_mm = e_ts_mm/max_values_ts_mm

#                 mse_diff_trig = (explloss.explloss_ssim(e_ts_om,e_ts_mm)) 
#                 mse_diff_trig_mean = (explloss.explloss_ssim(e_ts_om,e_ts_mm,'mean')) 
#                 #mse_diff_trig_list.append(mse_diff_trig.detach().cpu().numpy())  
        
#             print(str(nums+10)+" images done")
        

        
        

#         new_df = pd.DataFrame({

#                 'ground_truth' : ground_truth_str,

#                 'prediction_original_image_original_model' :preds_cs_om ,
#                 'probability_original_image_original_model': top_probs_cs_om,
#                 'predicted_class_name_original_image_original_model' : class_cs_om,

#                 'prediction_original_image_man_model': preds_cs_mm,
#                 'probability_original_image_man_model' : top_probs_cs_mm,
#                 'predicted_class_name_original_image_man_model' : class_cs_mm,

#                 'prediction_tri_image_original_model': p_ts_om,
#                 'probability_tri_image_original_model': top_probs_ts_om,
#                 'predicted_class_name_tri_image_original_model' : class_ts_om,

#                 'prediction_tri_image_man_model': p_ts_mm,
#                 'probability_tri_image_man_model': top_probs_cs_om,
#                 'predicted_class_name_tri_image_man_model' : class_ts_mm,

#                 'mse_diff': mse_diff.detach().cpu().numpy(),
#                 'mse_diff_tri':mse_diff_trig.detach().cpu().numpy(),
#                 'mse_diff_mean': mse_diff_mean.detach().cpu().numpy(),
#                 'mse_diff_tri_mean':mse_diff_trig_mean.detach().cpu().numpy()

#             })


#         new_df_all = pd.DataFrame({
#                 'all_probability_original_image_original_model': [all_top_probs_cs_om],
#                 'all_probability_original_image_man_model' : [all_top_probs_cs_mm],
#                 'all_probability_tri_image_original_model': [all_top_probs_ts_om],
#                 'all_probability_tri_image_man_model':[all_top_probs_ts_mm],
            
            
#             })    
                
            
#         # new_df.to_csv('/home/goad01/cvpr/output_'+str(exp_number)+'.csv')    
#         # new_df_all.to_csv('/home/goad01/cvpr/output_'+str(exp_number)+'_all.csv')
#         new_df.to_csv('/home/goad01/cvpr/output_'+str(exp_number)+'.csv', mode='a', header=False)
#         new_df_all.to_csv('/home/goad01/cvpr/output_'+str(exp_number)+'_all.csv', mode='a',header=False)
        








