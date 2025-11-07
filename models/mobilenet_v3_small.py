import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from enum import Enum

__all__ = ['MobileNetV3Small', 'mobilenet_v3_small']

# ---------- shared utils from your ResNet wrapper ----------

def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

def forward_hook(self, input, output):
    self.X = input[0]
    self.Y = output

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S, create_graph=False):
        C = torch.autograd.grad(Z, X, S, create_graph=create_graph, retain_graph=True)
        return C

    def relprop(self, R, alpha=1, create_graph=False):
        return R

class myAdd(RelProp):
    def forward(self, inputs):
        return torch.add(*inputs)
    def relprop(self, R, alpha=1, create_graph=False):
        d0 = safe_divide(self.X[0], self.Y)
        d1 = safe_divide(self.X[1], self.Y)
        return [torch.mul(R, d0), torch.mul(R.clone(), d1)]

class mySequential(nn.Sequential):
    def relprop(self, R, alpha=1, create_graph=False):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha, create_graph=create_graph)
        return R

class myClone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        return [input for _ in range(num)]
    def relprop(self, R, alpha=1, create_graph=False):
        return torch.add(R[0], R[1])

class myConv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        output_padding = self.X.size()[2] - (
            (self.Y.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding, groups=self.groups)

    def relprop(self, R, alpha=1, create_graph=False):
        if self.X.shape[1] == 3:
            # Z+ rule for first conv
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            H = self.X * 0 + torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) \
                 - torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) \
                 - torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) + 1e-9
            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            return C
        else:
            # Alpha-Beta rule
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z = Z1 + Z2
                S = safe_divide(R, Z)
                C1 = x1 * self.gradprop(Z1, x1, S, create_graph=create_graph)[0]
                C2 = x2 * self.gradprop(Z2, x2, S, create_graph=create_graph)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)
            return alpha * activator_relevances - beta * inhibitor_relevances

class myLinear(nn.Linear, RelProp):
    def relprop(self, R, alpha=1, create_graph=False):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            Z = Z1 + Z2
            S = safe_divide(R, Z)
            C1 = x1 * self.gradprop(Z1, x1, S, create_graph=create_graph)[0]
            C2 = x2 * self.gradprop(Z2, x2, S, create_graph=create_graph)[0]
            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)
        return alpha * activator_relevances - beta * inhibitor_relevances

class myBatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha=1, create_graph=False):
        X = self.X
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        return self.X * Ca

class myAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelProp):
    def relprop(self, R, alpha=1, create_graph=False):
        px = torch.clamp(self.X, min=0)
        def f(x1):
            Z1 = F.adaptive_avg_pool2d(x1, self.output_size)
            S1 = safe_divide(R, Z1)
            C1 = x1 * self.gradprop(Z1, x1, S1, create_graph=create_graph)[0]
            return C1
        return f(px)

# ---------- MobileNetV3-specific pieces ----------

def hard_sigmoid(x):
    return torch.clamp((x + 3.) / 6., 0., 1.)

def hard_swish(x):
    return x * hard_sigmoid(x)

class SEBlock(RelProp):
    """
    Squeeze-and-Excitation block.
    To keep parity with your wrapper style, we implement forward with my* parts.
    For relprop, we conservatively pass relevance to the input (shape-safe).
    """
    def __init__(self, in_ch, reduce_ratio=4):
        super().__init__()
        mid = max(1, in_ch // reduce_ratio)
        self.pool = myAdaptiveAvgPool2d(1)
        self.fc1 = myLinear(in_ch, mid)
        self.fc2 = myLinear(mid, in_ch)

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = hard_sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s

    def relprop(self, R, alpha=1, create_graph=False):
        # Simple, stable choice: pass relevance back unchanged to main input.
        # (Keeps interface identical to your other modules; avoids tricky product splits.)
        return R

class MobileBottleneck(nn.Module):
    """
    MobileNetV3 bottleneck (expansion -> depthwise -> SE? -> projection) with residual when possible.
    Uses your my* layers and activation_wrapper pattern.
    """
    def __init__(self, in_ch, out_ch, kernel, stride, expand_ch, use_se, activation_wrapper):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.use_se = use_se
        self.activation_wrapper = activation_wrapper
        padding = (kernel - 1) // 2

        # layers
        self.clone = myClone()
        self.add = myAdd()

        # 1x1 expand
        self.expand = None
        if expand_ch != in_ch:
            self.expand = nn.Sequential(
                myConv2d(in_ch, expand_ch, kernel_size=1, stride=1, padding=0, bias=False),
                myBatchNorm2d(expand_ch),
            )

        # depthwise
        self.depthwise = nn.Sequential(
            myConv2d(expand_ch, expand_ch, kernel_size=kernel, stride=stride, padding=padding, groups=expand_ch, bias=False),
            myBatchNorm2d(expand_ch),
        )

        # SE
        self.se = SEBlock(expand_ch) if use_se else nn.Identity()

        # project
        self.project = nn.Sequential(
            myConv2d(expand_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            myBatchNorm2d(out_ch),
        )

    def forward(self, x):
        if self.use_res:
            x1, x2 = self.clone(x, 2)
        else:
            x1 = x
        out = x1
        if self.expand is not None:
            out = self.expand[0](out)
            out = self.expand[1](out)
            out = self.activation_wrapper[0](out)
        out = self.depthwise[0](out)
        out = self.depthwise[1](out)
        out = self.activation_wrapper[0](out)
        out = self.se(out) if self.use_se else out
        out = self.project[0](out)
        out = self.project[1](out)
        if self.use_res:
            out = self.add([out, x2])
        return out

    def relprop(self, R, alpha, create_graph=False):
        if self.use_res:
            R, R_res = self.add.relprop(R, alpha, create_graph=create_graph)
            # residual is identity (no layer), so just return clone.relprop to merge at the end
            R = self.project[1].relprop(R, alpha, create_graph=create_graph)
            R = self.project[0].relprop(R, alpha, create_graph=create_graph)

            # SE
            if self.use_se:
                R = self.se.relprop(R, alpha, create_graph=create_graph)

            # depthwise + act
            R = self.depthwise[1].relprop(R, alpha, create_graph=create_graph)
            R = self.depthwise[0].relprop(R, alpha, create_graph=create_graph)

            # expand
            if self.expand is not None:
                R = self.expand[1].relprop(R, alpha, create_graph=create_graph)
                R = self.expand[0].relprop(R, alpha, create_graph=create_graph)

            # merge relevance from residual path
            R = self.clone.relprop([R, R_res], alpha, create_graph=create_graph)
            return R
        else:
            R = self.project[1].relprop(R, alpha, create_graph=create_graph)
            R = self.project[0].relprop(R, alpha, create_graph=create_graph)
            if self.use_se:
                R = self.se.relprop(R, alpha, create_graph=create_graph)
            R = self.depthwise[1].relprop(R, alpha, create_graph=create_graph)
            R = self.depthwise[0].relprop(R, alpha, create_graph=create_graph)
            if self.expand is not None:
                R = self.expand[1].relprop(R, alpha, create_graph=create_graph)
                R = self.expand[0].relprop(R, alpha, create_graph=create_graph)
            return R

# ---------- Top-level MobileNetV3Small wrapper (CIFAR-10) ----------

class ActivationMode(Enum):
    RELU = 1
    SOFTPLUS = 2

class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small for CIFAR-10 with your wrapper style:
      - my* layers + hooks for X/Y
      - activation_wrapper toggle (relu/softplus)
      - forward, forward_withoutfcl, forward_feature
      - relprop that mirrors your ResNet structure
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.activation_wrapper = [lambda x: torch.nn.functional.relu(x)]
        self.activationmode = ActivationMode.RELU

        # Stem: CIFAR-friendly (stride 1)
        self.conv_stem = myConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_stem = myBatchNorm2d(16)

        # Config (out_c, k, s, exp_c, se)
        cfg = [
            # in=16
            (16, 3, 2, 16,  True),   # -> 16x16
            (24, 3, 2, 72,  False),  # -> 8x8
            (24, 3, 1, 88,  False),
            (40, 5, 2, 96,  True),   # -> 4x4
            (40, 5, 1, 240, True),
            (40, 5, 1, 240, True),
            (48, 5, 1, 120, True),
            (48, 5, 1, 144, True),
            (96, 5, 2, 288, True),   # -> 2x2
            (96, 5, 1, 576, True),
            (96, 5, 1, 576, True),
        ]

        layers = []
        in_ch = 16
        for out_c, k, s, exp_c, se in cfg:
            layers.append(MobileBottleneck(in_ch, out_c, k, s, exp_c, se, self.activation_wrapper))
            in_ch = out_c
        self.blocks = mySequential(*layers)

        # Head
        self.conv_head = myConv2d(in_ch, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_head = myBatchNorm2d(576)
        self.avgpool = myAdaptiveAvgPool2d((1, 1))
        self.fc = myLinear(576, num_classes)

        self.apply(_weights_init)

    # activation switches like your ResNet
    def set_softplus(self, beta):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.softplus(x, beta=beta)
        self.activationmode = ActivationMode.SOFTPLUS

    def set_relu(self):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.relu(x)
        self.activationmode = ActivationMode.RELU

    # forwards
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = self.activation_wrapper[0](x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn_head(x)
        x = self.activation_wrapper[0](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_withoutfcl(self, x):
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = self.activation_wrapper[0](x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn_head(x)
        x = self.activation_wrapper[0](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_feature(self, x):
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = self.activation_wrapper[0](x)
        x = self.blocks(x)
        return x

    # relprop mirroring your ResNet flow
    def relprop(self, R, alpha, create_graph=False, break_at_basicblocks=False):
        R = self.fc.relprop(R, alpha, create_graph=create_graph)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.relprop(R, alpha, create_graph=create_graph)

        R = self.bn_head.relprop(R, alpha, create_graph=create_graph)
        R = self.conv_head.relprop(R, alpha, create_graph=create_graph)

        if break_at_basicblocks:
            return R

        R = self.blocks.relprop(R, alpha, create_graph=create_graph)

        R = self.bn_stem.relprop(R, alpha, create_graph=create_graph)
        R = self.conv_stem.relprop(R, alpha, create_graph=create_graph)
        return R

def mobilenet_v3_small(**kwargs):
    return MobileNetV3Small(**kwargs)

# quick sanity helper (matches your 'test' style, optional)
def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.cpu().numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

if __name__ == "__main__":
    print('mobilenet_v3_small')
    test(mobilenet_v3_small(num_classes=10))

# --------- Weight transfer from torchvision hub model (helper) ---------

def _collect_tv_layers_for_transfer(tv_model):
    """Collect a flat ordered list of (type, module) for conv/bn/se-conv layers in torchvision MNv3-Small.
    We walk tv_model.features sequentially and extract Conv2d/BatchNorm2d in encounter order,
    including SE 1x1 convs.
    """
    import torch.nn as nn
    layers = []
    def visit(m):
        # record only Conv2d and BatchNorm2d
        if isinstance(m, nn.Conv2d):
            layers.append(("conv", m))
        elif isinstance(m, nn.BatchNorm2d):
            layers.append(("bn", m))
        else:
            for c in m.children():
                visit(c)
    # Prefer traversing features if present
    root = getattr(tv_model, 'features', tv_model)
    for child in root.children():
        visit(child)
    # Also try final pre-classifier conv(+bn) if present outside features
    if hasattr(tv_model, 'conv'):  # some impls expose final conv as tv_model.conv
        visit(tv_model.conv)
    return layers

def _collect_local_layers_for_transfer(local_model):
    """Collect a flat ordered list of (type, module) for local model:
    - myConv2d => ("conv", module)
    - myBatchNorm2d => ("bn", module)
    - SEBlock.fc1/fc2 (Linear) => ("se_linear", module)
    The order follows the forward traversal from stem -> blocks -> head.
    """
    layers = []
    from torch import nn as _nn

    def visit(m):
        if isinstance(m, myConv2d):
            layers.append(("conv", m))
        elif isinstance(m, myBatchNorm2d):
            layers.append(("bn", m))
        elif isinstance(m, SEBlock):
            # order: fc1 then fc2 to match tv reduce/expand convs
            layers.append(("se_linear", m.fc1))
            layers.append(("se_linear", m.fc2))
        else:
            for c in m.children():
                visit(c)

    # Traverse explicitly in network order
    visit(local_model.conv_stem)
    visit(local_model.bn_stem)
    for blk in local_model.blocks:
        visit(blk)
    visit(local_model.conv_head)
    visit(local_model.bn_head)
    # exclude avgpool/fc here; tv classifier differs; we'll skip
    return layers

@torch.no_grad()
def transfer_from_torchvision_mnv3_small(local_model: MobileNetV3Small, tv_model) -> int:
    """Best-effort weight transfer from torchvision mobilenet_v3_small to our local wrapper.

    Strategy:
    - Flatten conv/bn (and SE 1x1 conv) layers from tv in order.
    - Flatten corresponding conv/bn (and SE linear) layers from local in order.
    - Copy matching shapes. For SE, reshape 1x1 conv weights (O, I, 1, 1) -> (O, I) for Linear.
    - Skip classifier layers; local head uses 576->num_classes directly.

    Returns the number of parameter tensors transferred.
    """
    tv_layers = _collect_tv_layers_for_transfer(tv_model)
    loc_layers = _collect_local_layers_for_transfer(local_model)

    transfers = 0
    i_tv = 0
    i_loc = 0
    while i_tv < len(tv_layers) and i_loc < len(loc_layers):
        ttype, tmod = tv_layers[i_tv]
        ltype, lmod = loc_layers[i_loc]

        if ttype == "conv" and ltype == "conv":
            # weights
            if hasattr(lmod, 'weight') and hasattr(tmod, 'weight') and lmod.weight.shape == tmod.weight.shape:
                lmod.weight.copy_(tmod.weight)
                transfers += 1
            # bias (rare for convs here)
            if getattr(lmod, 'bias', None) is not None and getattr(tmod, 'bias', None) is not None and lmod.bias.shape == tmod.bias.shape:
                lmod.bias.copy_(tmod.bias)
                transfers += 1
            i_tv += 1
            i_loc += 1
        elif ttype == "bn" and ltype == "bn":
            if lmod.weight.shape == tmod.weight.shape:
                lmod.weight.copy_(tmod.weight); transfers += 1
            if lmod.bias.shape == tmod.bias.shape:
                lmod.bias.copy_(tmod.bias); transfers += 1
            if lmod.running_mean.shape == tmod.running_mean.shape:
                lmod.running_mean.copy_(tmod.running_mean); transfers += 1
            if lmod.running_var.shape == tmod.running_var.shape:
                lmod.running_var.copy_(tmod.running_var); transfers += 1
            i_tv += 1
            i_loc += 1
        elif ttype == "conv" and ltype == "se_linear":
            # tv SE conv1x1 -> local Linear
            w_src = tmod.weight
            if w_src.ndim == 4 and w_src.shape[2:] == (1, 1) and lmod.weight.shape == (w_src.shape[0], w_src.shape[1]):
                lmod.weight.copy_(w_src.view(w_src.shape[0], w_src.shape[1]))
                transfers += 1
                # bias
                if getattr(tmod, 'bias', None) is not None and getattr(lmod, 'bias', None) is not None and lmod.bias.shape == tmod.bias.shape:
                    lmod.bias.copy_(tmod.bias); transfers += 1
                elif getattr(lmod, 'bias', None) is not None and lmod.bias is not None:
                    # tv conv had no bias -> zero init
                    lmod.bias.zero_()
            i_tv += 1
            i_loc += 1
        else:
            # Types do not align; try advancing the tv side (tv has extra activation-only wrappers etc.)
            i_tv += 1

    return transfers
