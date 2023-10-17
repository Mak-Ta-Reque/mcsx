import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, load_image, make_dir


def save_saliency_map(numpy_array, filename, cmap=None):
    # Convert the tensor to a numpy array and squeeze to remove the channel dimension
   
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image with a grayscale colormap
    if cmap is not None:
        print(cmap)
        ax.imshow(numpy_array, cmap=cmap)
    else:
        ax.imshow(numpy_array)

    # Remove the axis
    ax.axis('off')

    # Save the figure as a PDF
    fig.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
    

def heatmap_to_image(heatmap):
    """
    Helper image to convert torch tensor containing a heatmap into image.
    """
    if len(heatmap.shape) == 4:
        heatmap = heatmap.permute(0, 2, 3, 1)
    
    img = heatmap.squeeze().data.cpu().numpy()

    img = img / np.max(np.abs(img))  # divide by maximum
    img = np.maximum(-1, img)
    img = np.minimum(1, img) * 0.5  # clamp to -1 and divide by two -> range [-0.5, 0.5]
    img = img + 0.5

    return img
def torch_to_image(tensor, mean=0, std=1):
    """
    Helper function to convert torch tensor containing input data into image.
    """
    if len(tensor.shape) == 4:
        img = tensor.permute(0, 2, 3, 1)

    img = img.contiguous().squeeze().detach().cpu().numpy()

    img = img * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    return np.clip(img, 0, 1)



def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--img', type=str, default='../data/collie4.jpeg', help='image net file to run attack on')
    argparser.add_argument('--x', type=str, default='', help="tensor to calculate expl from")
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--output_dir', type=str, default='../output/', help='directory to save results to')
    argparser.add_argument('--betas', nargs='+', help='beta values for softplus explanations', type=float,
                           default=[10, 3, 1])
    argparser.add_argument('--method', help='algorithm for expls',
                           choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                    'pattern_attribution', 'grad_times_input', 'gradcam'],
                           default='gradient')
    args = argparser.parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    method = getattr(ExplainingMethod, args.method)

    # load model
    data_mean = np.array([0.485, 0.456, 0.406])
    data_std = np.array([0.229, 0.224, 0.225])
    vgg_model = torchvision.models.vgg16(pretrained=True)
    model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=None)
    if method == ExplainingMethod.pattern_attribution:
        model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
    model = model.eval().to(device)

    # load images
    x = load_image(data_mean, data_std, device, args.img)
    if len(args.x) > 0:
        x = torch.load(args.x).to(device)

    
    
    # produce expls
    expls = []
    expl, _, org_idx = get_expl(model, x, method)
    #expls.append(expl)
    #captions = ["Image", "Expl. with ReLU"]

    #for beta in args.betas:
    #    model.change_beta(beta)
    #    expl, _, _ = get_expl(model, x, method, desired_index=org_idx)
    #    expls.append(expl)
    #    captions.append(f'Expl. with softplus \nbeta={beta}')

    # save results
    output_dir = make_dir(args.output_dir)
    heatmap = heatmap_to_image(expl)
    save_saliency_map(heatmap, output_dir +'/explantion.pdf', cmap="viridis")
    
    print(x.size())
    x_array = torch_to_image(x, mean=data_mean, std=data_std)
    save_saliency_map(x_array, output_dir +'/input.pdf')
    
    plt.figure()
    plt.axis('off')
    # Display the image
    plt.imshow(x_array)

    # Display the heatmap with transparency
    plt.imshow(heatmap, alpha=0.5, cmap='viridis_r')

    # Save the figure as a PDF
    #plt.savefig(output_dir +'/saliency.pdf', format='pdf')
    plt.savefig(output_dir +'/saliency.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    # Create a figure and axis
    #fig, ax = plt.subplots()

# Display the image
    #ax.imshow(numpy_array, cmap='gray')

# Remove the axis
    #ax.axis('off')

# Save the figure as a PDF
    #fig.savefig(output_dir +'/explantion.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    
if __name__ == "__main__":
    main()
