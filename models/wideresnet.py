
# ---------------- WideResNet additions ----------------



'''
This is a heavily adopted ResNet implementation. The initial version was provided by Yerlan Idelbayev.
'''

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202', 'WideResNet', 'wideresnet28_10']

def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

def forward_hook(self, input, output):
    self.X = input[0]
    self.Y = output

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)
        #self.register_forward_pre_hook(forward_pre_hook)

    def gradprop(self, Z, X, S, create_graph=False):
        C = torch.autograd.grad(Z, X, S, create_graph=create_graph, retain_graph=True)
        return C

    def relprop(self, R, alpha = 1, create_graph=False):
        return R

class myAdd(RelProp):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha=1, create_graph=False):
        d0 = safe_divide(self.X[0],self.Y)
        d1 = safe_divide(self.X[1],self.Y)

        return [torch.mul(R,d0),torch.mul(R.clone(),d1)]

class myPadLayer(nn.Module):
    def __init__(self, planes):
        super(myPadLayer, self).__init__()
        self.planes = planes

    def forward(self, input):
        return nn.functional.pad(input[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)

    def relprop(self, R, alpha, create_graph=False):
        #R = R[:,self.planes//4:-(self.planes//4)] # Reduce padded planes
        #m = nn.Upsample(scale_factor=2, mode='nearest')

        #R = m(R)
        #R[:, :, 1::2, 1::2] = 0

        device = torch.device(os.getenv('CUDADEVICE'))
        B,C,W,H = R.shape
        Rnew = torch.zeros((B,C-self.planes//2,W*2,H*2)).to(device)
        Rnew[:, :, ::2, ::2] = R[:,self.planes//4:-(self.planes//4),:,:]
        return Rnew

class myConv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        output_padding = self.X.size()[2] - (
                (self.Y.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha = 1, create_graph=False):
        if self.X.shape[1] == 3:
            # Apply Zeta Rule
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C

        else:
            # Apply normal Alpha Beta Rule
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

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R

class myLinear(nn.Linear, RelProp):

    def relprop(self, R, alpha = 1, create_graph=False):
        # Apply Alpha Beta Rule
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

        out = alpha * activator_relevances - beta * inhibitor_relevances

        return out

class myBatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha = 1, create_graph=False):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R

class myClone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha = 1, create_graph=False):
        return torch.add(R[0],R[1])

class mySequential(nn.Sequential):
    def relprop(self, R, alpha = 1, create_graph=False):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha, create_graph=create_graph)
        return R

class myAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelProp):
    def relprop(self, R, alpha=1, create_graph=False):
        px = torch.clamp(self.X, min=0)

        def f(x1):
            Z1 = F.adaptive_avg_pool2d(x1, self.output_size)
            S1 = safe_divide(R, Z1)
            C1 = x1 * self.gradprop(Z1, x1, S1, create_graph=create_graph)[0]
            return C1

        return f(px)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, activation_wrapper, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.activation_wrapper = activation_wrapper
        self.clone = myClone()
        self.conv1 = myConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = myBatchNorm2d(planes)
        self.conv2 = myConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = myBatchNorm2d(planes)

        self.add = myAdd()

        self.shortcut = mySequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = myPadLayer(planes)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     myConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     myBatchNorm2d(self.expansion * planes)
                )

    def forward(self, input):
        x1, x2 = self.clone(input,2)
        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.add([out, self.shortcut(x2)])
        out = self.activation_wrapper[0](out)

        return out

    def relprop(self, relevances, alpha, create_graph=False):

        relevances, relevances2 = self.add.relprop(relevances, alpha, create_graph=create_graph)
        relevances2 = self.shortcut.relprop(relevances2, alpha, create_graph=create_graph)
        relevances = self.bn2.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.conv2.relprop(relevances, alpha, create_graph=create_graph)

        relevances = self.bn1.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.conv1.relprop(relevances, alpha, create_graph=create_graph)
        return self.clone.relprop([relevances, relevances2], alpha, create_graph=create_graph)

from enum import Enum
class ActivationMode (Enum):
    RELU = 1
    SOFTPLUS = 2


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.activation_wrapper = [lambda x: torch.nn.functional.relu(x)] # This can be changed dynamically
        self.activationmode = ActivationMode.RELU
        self.in_planes = 16

        self.conv1 = myConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = myBatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = myAdaptiveAvgPool2d((1, 1))
        self.linear = myLinear(64, num_classes)

        self.apply(_weights_init)

    def set_softplus(self, beta):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.softplus(x, beta=beta)
        self.activationmode = ActivationMode.SOFTPLUS

    def set_relu(self):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.relu(x)
        self.activationmode = ActivationMode.RELU

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.activation_wrapper, stride))
            self.in_planes = planes * block.expansion

        return mySequential(*layers)

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_withoutfcl(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out
    
    def forward_feature(self, input):
            out = self.conv1(input)
            out = self.bn1(out)
            out = self.activation_wrapper[0](out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            return out

    def relprop(self, relevances, alpha, create_graph=False, break_at_basicblocks=False):
        relevances = self.linear.relprop(relevances, alpha, create_graph=create_graph)
        relevances = relevances.reshape_as(self.avgpool.Y)
        relevances = self.avgpool.relprop(relevances, alpha, create_graph=create_graph)

        if break_at_basicblocks:
            return relevances

        relevances = self.layer3.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.layer2.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.layer1.relprop(relevances, alpha, create_graph=create_graph)

        relevances = self.bn1.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.conv1.relprop(relevances, alpha, create_graph=create_graph)
        return relevances

def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])

def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

class WideBasic(nn.Module):
    """WideResNet basic block using your my* layers and LRP-friendly structure.
       Layout: Conv3x3 -> BN -> Act -> (optional Dropout) -> Conv3x3 -> BN -> Add(shortcut) -> Act
    """
    expansion = 1

    def __init__(self, in_planes, planes, activation_wrapper, stride=1, dropRate=0.0):
        super(WideBasic, self).__init__()
        self.activation_wrapper = activation_wrapper
        self.dropRate = float(dropRate)

        self.clone = myClone()
        self.conv1 = myConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = myBatchNorm2d(planes)
        self.conv2 = myConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = myBatchNorm2d(planes)

        # Option B style projection when shape changes (typical for WRN)
        self.shortcut = mySequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = mySequential(
                myConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                myBatchNorm2d(self.expansion * planes)
            )

        self.add = myAdd()
        self.dropout = nn.Dropout(p=self.dropRate) if self.dropRate > 0.0 else nn.Identity()

    def forward(self, x):
        x_main, x_res = self.clone(x, 2)
        out = self.conv1(x_main)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.add([out, self.shortcut(x_res)])
        out = self.activation_wrapper[0](out)
        return out

    def relprop(self, R, alpha, create_graph=False):
        # Split relevance across residual add
        R_main, R_res = self.add.relprop(R, alpha, create_graph=create_graph)
        # Shortcut path
        R_res = self.shortcut.relprop(R_res, alpha, create_graph=create_graph)
        # Main path (note: dropout is treated as identity for relprop)
        R_main = self.bn2.relprop(R_main, alpha, create_graph=create_graph)
        R_main = self.conv2.relprop(R_main, alpha, create_graph=create_graph)

        R_main = self.bn1.relprop(R_main, alpha, create_graph=create_graph)
        R_main = self.conv1.relprop(R_main, alpha, create_graph=create_graph)

        return self.clone.relprop([R_main, R_res], alpha, create_graph=create_graph)


class WideResNet(nn.Module):
    """WideResNet for CIFAR-style inputs using your custom layers and LRP pipeline.
       depth: 6n+4  (e.g., 28 -> n=4)
       widen_factor: K (e.g., 10 for WRN-28-10)
    """
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth must be 6n+4 (e.g., 16, 22, 28, 40)."
        n = (depth - 4) // 6

        # keep same activation wrapper scheme as your ResNet
        self.activationmode = ActivationMode.RELU
        self.activation_wrapper = [lambda x: torch.nn.functional.relu(x)]

        widths = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        self.in_planes = widths[0]
        self.conv1 = myConv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = myBatchNorm2d(widths[0])

        self.layer1 = self._make_group(WideBasic, widths[1], n, stride=1, dropRate=dropRate)
        self.layer2 = self._make_group(WideBasic, widths[2], n, stride=2, dropRate=dropRate)
        self.layer3 = self._make_group(WideBasic, widths[3], n, stride=2, dropRate=dropRate)

        self.avgpool = myAdaptiveAvgPool2d((1, 1))
        self.linear  = myLinear(widths[3], num_classes)

        self.apply(_weights_init)

    def set_softplus(self, beta):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.softplus(x, beta=beta)
        self.activationmode = ActivationMode.SOFTPLUS

    def set_relu(self):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.relu(x)
        self.activationmode = ActivationMode.RELU

    def _make_group(self, block, planes, num_blocks, stride, dropRate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, self.activation_wrapper, stride=s, dropRate=dropRate))
            self.in_planes = planes * block.expansion
        return mySequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_withoutfcl(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward_feature(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

    def relprop(self, R, alpha, create_graph=False, break_at_basicblocks=False):
        R = self.linear.relprop(R, alpha, create_graph=create_graph)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.relprop(R, alpha, create_graph=create_graph)

        if break_at_basicblocks:
            return R

        R = self.layer3.relprop(R, alpha, create_graph=create_graph)
        R = self.layer2.relprop(R, alpha, create_graph=create_graph)
        R = self.layer1.relprop(R, alpha, create_graph=create_graph)

        R = self.bn1.relprop(R, alpha, create_graph=create_graph)
        R = self.conv1.relprop(R, alpha, create_graph=create_graph)
        return R


# --------- convenience factory (common setting for CIFAR) ----------
def wideresnet_28_10(num_classes=10, dropRate=0.0, *, depth: int = 28, widen_factor: int = 10):
    """Readable factory for WideResNet 28-10.

    Using underscores (wideresnet_28_10) improves clarity vs concatenated name.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    dropRate : float
        Dropout probability applied in WideBasic blocks.
    depth : int, optional (keyword-only)
        Network depth (must satisfy 6n+4). Defaults to 28.
    widen_factor : int, optional (keyword-only)
        Channel widening factor. Defaults to 10.

    Returns
    -------
    WideResNet
        Instantiated WideResNet model.
    """
    return WideResNet(depth=depth, widen_factor=widen_factor, num_classes=num_classes, dropRate=dropRate)

# Backwards-compatible alias (original name kept so existing code keeps working)
def wideresnet28_10(*args, **kwargs):
    return wideresnet_28_10(*args, **kwargs)
