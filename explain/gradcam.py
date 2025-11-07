import os, math
import torch
import torch.nn.functional as F
from torch import nn

def gradcam(model, samples, create_graph=False, res_to_explain=None):
    device = torch.device(os.getenv('CUDADEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    samples = samples.to(device)

    activationmap = None
    gradients = None

    def _encode_one_hot(y, ids):
        one_hot = torch.zeros_like(y, device=y.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward_hook(module, input_, output):
        nonlocal activationmap
        activationmap = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach().clone()

    # -------- pick target layer ----------
    modeltype = os.getenv("MODELTYPE", "")
    target_layer = None
    hook_module = None

    if modeltype in {'resnet20_normal','resnet20_gtsrb_normal','gtsrb_resnet20','cifar10_resnet20'}:
        target_layer = 'layer3.2.bn2'
    elif modeltype in {'resnet20_nbn','resnet20_freeze_bn'}:
        target_layer = 'layer3.2.conv2'
    elif modeltype == 'vgg13_normal':
        target_layer = 'features.22'
    elif modeltype in {'vgg13bn_normal', 'gtsrb_vgg13bn', 'cifar10_vgg13bn'}:
        target_layer = 'features.24'
    elif modeltype in {'cifar10_wideresnet28_10','gtsrb_wideresnet28_10'}:
        target_layer = 'layer3.1.bn2'
    elif modeltype in {'cifar10_mobilenetv3small','gtsrb_mobilenetv3small'}:
        target_layer = 'bn_head'
    elif modeltype in {'cifar10_vit_b_16','gtsrb_vit_b_16'}:
        # robustly find the LAST encoder block's norm1 without relying on exact string names
        # works with torchvision VisionTransformer
        last_norm1 = None
        for m in model.modules():
            # Encoder blocks in torchvision have attributes like norm1/attn/norm2/mlp
            if hasattr(m, 'norm1') and isinstance(getattr(m, 'norm1'), nn.LayerNorm):
                last_norm1 = m.norm1
        if last_norm1 is None:
            raise RuntimeError("Could not locate ViT encoder block norm1 to hook.")
        hook_module = last_norm1
    else:
        raise Exception(f"No target layer specified for modeltype {modeltype}")

    # If we didn’t set hook_module directly (non-ViT), resolve by name
    if hook_module is None:
        for name, module in model.named_modules():
            if name == target_layer:
                hook_module = module
                break
        if hook_module is None:
            raise Exception(f"Target Layer {target_layer} not found!")

    hf = hook_module.register_forward_hook(forward_hook)
    hb = hook_module.register_full_backward_hook(backward_hook)

    try:
        samples.grad = None
        y = model(samples)

        prediction_ids = (y.argmax(dim=1).unsqueeze(1) if res_to_explain is None
                          else res_to_explain.unsqueeze(1))
        one_hot = _encode_one_hot(y, prediction_ids)

        y.backward(gradient=one_hot, create_graph=create_graph)

        # ------- CNN path (activations 4D) -------
        if activationmap.dim() == 4:
            # (B,C,H,W) and grads same shape
            weights = F.adaptive_avg_pool2d(gradients, 1)              # (B,C,1,1)
            gcam = (activationmap * weights).sum(dim=1, keepdim=True)  # (B,1,H,W)
            gcam = F.relu(gcam)
            gcam = F.interpolate(gcam, samples.shape[2:], mode="bilinear", align_corners=False)

        # ------- ViT path (activations 3D tokens) -------
        elif activationmap.dim() == 3:
            # activationmap, gradients: (B, N, C)
            B, N, C = activationmap.shape
            # Grad-CAM weights: average over tokens
            weights = gradients.mean(dim=1)            # (B, C)
            # token importance: sum over channels
            cam_tokens = torch.einsum('bnc,bc->bn', activationmap, weights)  # (B, N)

            # drop CLS token (index 0), reshape patches to SxS
            cam_patches = cam_tokens[:, 1:]           # (B, N-1)
            S = int(math.sqrt(cam_patches.shape[1]))
            if S * S != cam_patches.shape[1]:
                raise RuntimeError(f"Cannot reshape tokens to square: got {cam_patches.shape[1]} patches")
            cam = cam_patches.view(B, 1, S, S)        # (B,1,S,S)
            gcam = F.relu(cam)
            gcam = F.interpolate(gcam, size=samples.shape[2:], mode="bilinear", align_corners=False)

        else:
            raise RuntimeError(f"Unexpected activation shape: {activationmap.shape}")

        res = y.argmax(-1)
        return gcam, res, y

    finally:
        hf.remove()
        hb.remove()
