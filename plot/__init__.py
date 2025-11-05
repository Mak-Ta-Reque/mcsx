# System
import os
import pathlib

# Lib
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# Our source
import explain
from explain import *
import utils
from mcdropout import *
from torch import nn


def agreegate_prediction(prediction):
    votes = Counter(prediction).most_common(1)[0][0]
    return votes


def argmax_histogram(data, bins):
    data = data.cpu().detach().numpy()
    counts, bin_edges = np.histogram(data, bins)
    max_bin_index = np.argmax(counts)
    max_bin = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])
    return max_bin

def histogram_vote_one_batch(heatmaps):
    bins = int(torch.ceil(torch.log2(torch.tensor(heatmaps.size()[0])))) #//200
    #bins = heatmaps.size()[0]//2
    siz = heatmaps.size()
    his_res = torch.zeros(siz[-3], siz[-2], siz[-1])  
    for x in range(his_res.size()[-3]):
        for y in range(his_res.shape[-2]):
            for z in  range(his_res.shape[-1]):
                his_res[x, y,z] = argmax_histogram(heatmaps[:, x, y, z], bins)
    
    return his_res

def max_hist(heatmaps):
    heatmaps =  heatmaps.permute(1, 0, 2, 3, 4)
    f = torch.stack([histogram_vote_one_batch(it) for it in heatmaps], dim=0)
    return f

def get_grid(rows, cols, double_rows):
    """
    TODO describe
    :param rows:
    :param cols:
    :param double_rows: ?
    """
    fig = plt.figure()
    gs = fig.add_gridspec(rows, cols)
    double_rows = set(double_rows)
    axs = [None for _ in range(rows)]
    for row in range(rows):
        axs[row] = [None for _ in range(cols)]
        if row in double_rows:
            continue
        for col in range(cols):
            axs[row][col] = fig.add_subplot(gs[row, col])
    return fig, axs

def add_left_text(axs, text):
    """
    TODO describe
    :param axs:
    :param text:
    """
    if text is not None:
        pos = 0
        i = 0
        while pos < len(axs):
            a = axs[pos][0]
            if a is None:
                pos+=1
                continue
            a.text(-1, .5, text[i], transform=a.transAxes, fontsize=8)
            i += 1
            pos += 1

class RBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(RBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False, track_running_stats=False)

    def forward(self, x):
        return self.bn(x)

class CHBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(CHBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False, track_running_stats=False)

    def forward(self, X):
           # Calculate the mean and standard deviation of the tensor along the color channel
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + 1e-5)
        return X_hat

def replace_bn(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # If current module is a container, recurse on children
            model._modules[name] = replace_bn(module)

        if isinstance(module, nn.BatchNorm2d):
            # Use num_features from the original layer
            num_features = module.num_features
            model._modules[name] = CHBatchNorm(num_features).to(os.environ['CUDADEVICE'] )
    return model

def bnabdul_eval(model, input_data, explanation_method, create_graph=False, nsim = 20, hist=True):
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
    model = replace_bn(model)
    e, p, y = explain.explain_multiple(model, input_data, explanation_method=explanation_method, create_graph=create_graph)  
    return e, p, y

def buffabdul_eval(model, input_data, explanation_method, x_buff, create_graph=False, nsim = 20, hist=True):
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
    input_data = torch.cat((x_buff, input_data), dim=0)
    n_sim=nsim

    if not model.training:
        model.train()  # Set the model to training mode to enable dropout
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
    # Create an agreegator funtion for accuracy and predicton 
    if hist:
        mean_result_e  = max_hist(monte_carlo_results_e)[x_buff.shape[0]:]
        
    else:
        mean_result_e = monte_carlo_results_e.mean(dim=0)[x_buff.shape[0]:]
    std_deviation_e = monte_carlo_results_e.std(dim=0)
    mean_result_p = agreegate_prediction(monte_carlo_results_p)[x_buff.shape[0]:]
    mean_result_y = monte_carlo_results_y.mean(dim=0)[x_buff.shape[0]:]
    std_deviation_y = monte_carlo_results_y.std(dim=0)
    print(mean_result_e.shape)
    return mean_result_e, mean_result_p, mean_result_y

def bn_eval(model, input_data, explanation_method, create_graph=False, nsim = 0, hist=True):
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
    if not model.training:
        model.train()  # Set the model to training mode to disable batchnorm
 

    #with torch.no_grad():
    
    e, p, y = explain.explain_multiple(model, input_data, explanation_method=explanation_method, create_graph=create_graph)
    
    model.eval()

    
    return e, p, y

def abdul_eval(model, input_data, explanation_method, create_graph=False, nsim = 20, hist=True):
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
    n_sim=nsim
    if not model.training:
        model.train()  # Set the model to training mode to enable dropout
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
    # Create an agreegator funtion for accuracy and predicton 
    if hist:
        mean_result_e  = max_hist(monte_carlo_results_e)
        
    else:
        mean_result_e = monte_carlo_results_e.mean(dim=0)
    std_deviation_e = monte_carlo_results_e.std(dim=0)
    mean_result_p = agreegate_prediction(monte_carlo_results_p)
    mean_result_y = monte_carlo_results_y.mean(dim=0)
    std_deviation_y = monte_carlo_results_y.std(dim=0)
    
    return mean_result_e, mean_result_p, mean_result_y

def calculate_accuracy(outdir : pathlib.Path, epoch : int, original_model, manipulated_model, x_test : torch.Tensor, label_test : torch.Tensor, run, agg='max', save=True, show=False):
    """
    Creates an overview plot with clean samples, their version with trigger, explanations and predictions
    and explanations and predictions of the original models, as well as from the manipulated model. The number of
    samples depends on the number of attacks included in the run.

    :param outdir:
    :param epoch:
    :param original_model:
    :param manipulated_model:
    :param x_test:
    :param label_test:
    :param run:
    :param save:
    :param show:
    """
    robust_batch_size = 20
    # Choose samples
    samples = copy.deepcopy(x_test.detach().clone())
    ground_truth = label_test.detach().clone()
    if os.getenv("DATASET") == 'cifar10':
        ground_truth_str = [utils.cifar_classes[x] for x in ground_truth]
    elif os.getenv("DATASET") == 'gtsrb':
        ground_truth_str = [utils.gtsrb_classes[x] for x in ground_truth]
    else:
        ground_truth_str = f"no labels for {os.getenv('DATASET')}"


    manipulators = run.get_manipulators()
    num_explanation_methods = len(run.explanation_methodStrs)

    def postprocess_expls(expls):
        return utils.aggregate_explanations(agg, expls)

    trg_samples = []
    for manipulator in manipulators:
        ts = manipulator(copy.deepcopy(samples.detach().clone()))
        trg_samples.append(ts)

    trg_samples = torch.stack(trg_samples)
    for i in range(len(run.explanation_methodStrs)):
        #explanation_method = run.get_explanation_method(i)
        # Calculate the accuracy of clean data on clean model without mc_dropout
        original_model.eval()
        manipulated_model.eval()
        fresh_acc = acc(original_model,  samples, label_test)
        print("Non-robust accuracy on fresh model and fresh input: ",fresh_acc) # Frsh mode fresh input
        fresh_acc = acc(manipulated_model,  samples, label_test) # Attcked model fresh input
        print("Non-robust accuracy on targeted model and fresh input: ",fresh_acc) 
        fresh_acc = mc_acc(original_model,  samples, label_test, robust_batch_size) #
        print("Robust accuracy on fresh model and fresh input: ",fresh_acc) # 
        fresh_acc = mc_acc(manipulated_model,  samples, label_test, robust_batch_size) #
        print("Robust accuracy on targeted model and fresh input: ",fresh_acc) # 
        
        for man_id in range(run.num_of_attacks):
            original_model.eval()
            normal_acc = acc(original_model, trg_samples[man_id], label_test)
            print("Non-robuast accuracy on fresh model and targeted sample: ", normal_acc)
            mc_ac = mc_acc(original_model, trg_samples[man_id], label_test, robust_batch_size)
            print("Robust accuracy on n fresh model and targeted sample: ",  mc_ac)
       
        for man_id in range(run.num_of_attacks):
            manipulated_model.eval()
            ac_at_ex__acc = acc(manipulated_model, trg_samples[man_id], label_test)
            print(f"Non-robust accuracy on attacked model and targeted sample: ", ac_at_ex__acc)
          
            mc_ac = mc_acc(manipulated_model, trg_samples[man_id], label_test, robust_batch_size)
            print(f"Robust accuracy on attacked and targeted sample : ", mc_ac)
            
            
           






def plot_heatmaps(outdir : pathlib.Path, epoch : int, original_model, manipulated_model, x_test : torch.Tensor, label_test : torch.Tensor, run, agg='max', save=True, show=False, robust=False):
    """
    Creates an overview plot with clean samples, their version with trigger, explanations and predictions
    and explanations and predictions of the original models, as well as from the manipulated model. The number of
    samples depends on the number of attacks included in the run.

    :param outdir:
    :param epoch:
    :param original_model:
    :param manipulated_model:
    :param x_test:
    :param label_test:
    :param run:
    :param save:
    :param show:
    """
    
    # Store some clean data in the buffer
    indices = torch.randint(0, x_test.shape[0], (100,))
    x_buff =   copy.deepcopy(x_test[indices].detach().clone())
    
    if robust:
        explainer = abdul_eval
    else:
        explainer = explain.explain_multiple
        
    num_samples = 100
    st = 5005
    # Choose samples
    samples = copy.deepcopy(x_test[st:num_samples+st].detach().clone())
    ground_truth = label_test[st:num_samples+st].detach().clone()
    #print(os.getenv("DATASET"))
    if os.getenv("DATASET") == 'cifar10':
        ground_truth_str = [utils.cifar_classes[x] for x in ground_truth]
    elif os.getenv("DATASET") == 'gtsrb':
        ground_truth_str = [utils.gtsrb_classes[x] for x in ground_truth]
    else:
        ground_truth_str = f"no labels for {os.getenv('DATASET')}"


    manipulators = run.get_manipulators()
    num_explanation_methods = len(run.explanation_methodStrs)

    def postprocess_expls(expls):
        return utils.aggregate_explanations(agg, expls)

    trg_samples = []
    for manipulator in manipulators:
        ts = manipulator(copy.deepcopy(samples.detach().clone()))
        trg_samples.append(ts)

    trg_samples = torch.stack(trg_samples)
    
    expls = []
    expls_man = []

    trg_expls = []
    trg_preds = []
    trg_ys = []

    trg_expls_man = []
    trg_preds_man = []
    trg_ys_man = []
    for i in range(len(run.explanation_methodStrs)):
        explanation_method = run.get_explanation_method(i)
        # Generate the explanations of clean samples on the original model
        tmp, preds, ys = explain.explain_multiple(original_model, samples, explanation_method=explanation_method, create_graph=False)
        # preds and ys does not change for different explanations methods
        tmp = postprocess_expls(tmp)
        expls.append(tmp.detach())

        tmp, preds_man, ys_man = explain.explain_multiple(manipulated_model, samples, explanation_method=explanation_method, create_graph=False)
        # preds_man and ys_man does not change for different explanation methods
        tmp = postprocess_expls(tmp)
        expls_man.append(tmp.detach())

        # Generate explanation for the trigger samples in the original model
        tmp_expls = []
        tmp_preds = []
        tmp_ys = []
        for man_id in range(run.num_of_attacks):
            e, p, y = explain.explain_multiple(original_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
            #e, p, y  = abdul_eval(model = original_model, explantion_method = explanation_method, input_data =  trg_samples[man_id], n_sim=20)
            e = postprocess_expls(e)
            tmp_expls.append(e)
            tmp_preds.append(p)
            tmp_ys.append(y)

        tmp_expls = torch.stack(tmp_expls).detach()
        tmp_preds = torch.stack(tmp_preds)
        tmp_ys = torch.stack(tmp_ys)

        trg_expls.append(tmp_expls)
        trg_preds.append(tmp_preds)
        trg_ys.append(tmp_ys)


        # Generate explanation the trigger samples in the manipulated model
        tmp_expls_man = []
        tmp_preds_man = []
        tmp_ys_man = []
        for man_id in range(run.num_of_attacks):
            #e, p, y = explain.explain_multiple(manipulated_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
            e, p, y  = explainer(manipulated_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
            #e, p, y  = buffabdul_eval(manipulated_model, trg_samples[man_id], explanation_method=explanation_method, x_buff = x_buff, create_graph=False)
            e = postprocess_expls(e)
            tmp_expls_man.append(e)
            tmp_preds_man.append(p)
            tmp_ys_man.append(y)

        tmp_expls_man = torch.stack(tmp_expls_man).detach()
        tmp_preds_man = torch.stack(tmp_preds_man)
        tmp_ys_man = torch.stack(tmp_ys_man)

        trg_expls_man.append(tmp_expls_man)
        trg_preds_man.append(tmp_preds_man)
        trg_ys_man.append(tmp_ys_man)
    num_samples = 3
    num_images_per_sample = run.num_of_attacks + 1
    num_columns = num_samples * num_images_per_sample
    num_rows = 2 + (2 + (2*num_explanation_methods))

    fig, axs = get_grid(num_rows, num_columns, [1, 2+num_explanation_methods, 3+(2*num_explanation_methods)])
    axs = np.array(axs)
    fig.set_size_inches(2 + num_columns, num_rows)
    for i in range(num_samples):
        plot_single_sample(samples[i].cpu(), axs[0,i * num_images_per_sample], normalize=True)
    for i in range(num_samples):
        for man_id in range(run.num_of_attacks):
            plot_single_sample(trg_samples[man_id,i].cpu(), axs[0,i * num_images_per_sample + man_id + 1], normalize=True)

    alpha = 0.3

    # Write prediction on plot
    for i in range(num_samples):
        # Predictions for clean samples
        fig.text(0, -0.5, ground_truth_str[i], transform=axs[0][i * num_images_per_sample].transAxes)

        title0 = utils.top_probs_as_string(ys[i])
        # axs[1][i].text(0, 1.5, title0, transform=axs[1][i].transAxes, fontsize=8)
        fig.text(0, -1, title0, transform=axs[1+num_explanation_methods][i * num_images_per_sample].transAxes)

        title1 = utils.top_probs_as_string(ys_man[i])
        fig.text(0, -1, title1, transform=axs[2 + (2*num_explanation_methods)][i * num_images_per_sample].transAxes)

        # Predictions for manipulated samples
        for man_id in range(run.num_of_attacks):
            fig.text(0, -0.5, ground_truth_str[i], transform=axs[0][i * num_images_per_sample + man_id + 1].transAxes)

            title0 = utils.top_probs_as_string(trg_ys[0][man_id,i])
            # axs[1][i].text(0, 1.5, title0, transform=axs[1][i].transAxes, fontsize=8)
            fig.text(0, -1, title0, transform=axs[1+num_explanation_methods][i * num_images_per_sample + man_id + 1].transAxes)

            title1 = utils.top_probs_as_string(trg_ys_man[0][man_id,i])
            fig.text(0, -1, title1, transform=axs[2 + (2*num_explanation_methods)][i * num_images_per_sample + man_id + 1].transAxes) # +1 as the first 'manipulator' is no manipulator but clean

    # Plot the actual explanations
    for explId in range(num_explanation_methods):
        rowClean = 2 + explId
        rowMan = 3 + num_explanation_methods + explId
        for i in range(num_samples):
            # Plot explanations for clean samples
            plot_single_sample(expls[explId][i].cpu(), axs[rowClean, i * num_images_per_sample], cmap='plasma')
            plot_single_sample(samples[i].cpu(), axs[rowClean, i * num_images_per_sample], normalize=True, bw=True, alpha=alpha)
            # Different visualization: Input times Relevance
            #plot_single_sample((expls[i].repeat(3, 1, 1) * samples[i]).cpu(), axs[2, i * num_images_per_sample], cmap='plasma')

            plot_single_sample(expls_man[explId][i].cpu(), axs[rowMan, i * num_images_per_sample], cmap='plasma')
            plot_single_sample(samples[i].cpu(), axs[rowMan, i * num_images_per_sample], normalize=True, bw=True, alpha=alpha)

            # Plot explanations for trigger samples
            for man_id in range(run.num_of_attacks):
                plot_single_sample(trg_expls[explId][man_id,i].cpu(), axs[rowClean, i * num_images_per_sample + man_id + 1], cmap='plasma')
                plot_single_sample(trg_samples[man_id,i].cpu(), axs[rowClean, i * num_images_per_sample + man_id + 1], normalize=True, bw=True, alpha=alpha)

                plot_single_sample(trg_expls_man[explId][man_id,i].cpu(), axs[rowMan, i * num_images_per_sample + man_id + 1], cmap='plasma')
                plot_single_sample(trg_samples[man_id,i].cpu(), axs[rowMan, i * num_images_per_sample + man_id + 1], normalize=True, bw=True, alpha=alpha)

    rownames = ["Input"]
    for explId in range(num_explanation_methods):
        rownames.append( f'Orig. M. \n{run.explanation_methodStrs[explId]}')
    for explId in range(num_explanation_methods):
        rownames.append(f'Man. M. \n{run.explanation_methodStrs[explId]}')
    add_left_text(axs, rownames)
    plt.suptitle(run.get_params_str_row() + f' epoch {epoch}')

    # fig.tight_layout(pad=0, h_pad=0.5)
    if save:
        utils.save_multiple_formats(fig, outdir / f'plot_{epoch:03d}')
    if show:
        fig.show()
    if save:
        plt.close(fig)
    else:
        return fig

def plot_explanation_to_ax(relevances, sample, ax):
    """
    Return the figure of an explanation plotted on the sample.
    """
    ax.imshow(relevances.permute(1, 2, 0).detach().clone().cpu(), cmap='viridis', interpolation=None)
    ax.imshow(utils.unnormalize_images(sample).mean(dim=0, keepdim=True).permute(1, 2, 0).detach().clone().cpu(), cmap='gray', interpolation=None, alpha=0.4)
    ax.axis('off')

def plot_explanation(relevances, sample, ax=None):
    """
    Return the figure of an explanation plotted on the sample.
    """

    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    plt.tight_layout()
    plot_explanation_to_ax(relevances, sample, ax)
    return fig

def plot_sample(sample):
    """
    Plots a samples only
    """
    print(sample.shape)
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    plt.tight_layout()
    ax.set_axis_off()
    ax.imshow(sample.permute(1, 2, 0).detach().clone().cpu())
    ax.axis('off')
    return fig

def plot_single_sample(sample, ax, normalize=False, bw=False, cmap='gray', alpha=1.0):
    if normalize:
        sample = utils.unnormalize_images(sample.unsqueeze(0))[0]
    if bw:
        sample = sample.mean(dim=0,keepdim=True)

    ax.axis('off')
    ax.imshow(sample.permute(1, 2, 0), interpolation='none', cmap=cmap, alpha=alpha)

