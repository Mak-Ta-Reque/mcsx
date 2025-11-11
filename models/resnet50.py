"""Custom ResNet-50 implementation with LRP-compatible layers.

This module mirrors the structure of :mod:`models.resnet` but targets the
ImageNet-scale architecture (Bottleneck residual blocks). It provides helpers to
instantiate the model and load weights either from Torch Hub (pretrained on
ImageNet) or from local checkpoints used for manipulated models.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = [
    "ResNet",
    "resnet18",
    "resnet50",
    "load_imagenet_resnet18_model",
    "load_imagenet_resnet50_model",
    "load_imagenet_resnet18_model_local",
    "load_imagenet_resnet50_model_local",
    "load_imagenet_resnet18_manipulated",
    "load_imagenet_resnet50_manipulated",
]


def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())


def forward_hook(self: nn.Module, input: Any, output: torch.Tensor) -> None:
    self.X = input[0]
    self.Y = output


def _weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)


class RelProp(nn.Module):
    def __init__(self) -> None:
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S, create_graph=False):
        C = torch.autograd.grad(Z, X, S, create_graph=create_graph, retain_graph=True)
        return C

    def relprop(self, R, alpha=1, create_graph=False):  # noqa: ARG002
        return R


class myAdd(RelProp):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
        d0 = safe_divide(self.X[0], self.Y)
        d1 = safe_divide(self.X[1], self.Y)
        return [torch.mul(R, d0), torch.mul(R.clone(), d1)]


class myClone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
        return torch.add(R[0], R[1])


class mySequential(nn.Sequential):
    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha, create_graph=create_graph)
        return R


class myConv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            H = self.X * 0 + torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding)
            Za = Za - torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding)
            Za = Za - torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9
            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
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
    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
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
    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        weight = weight / (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5)
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * Ca
        return R


class myAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelProp):
    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
        px = torch.clamp(self.X, min=0)

        def f(x1):
            Z1 = F.adaptive_avg_pool2d(x1, self.output_size)
            S1 = safe_divide(R, Z1)
            C1 = x1 * self.gradprop(Z1, x1, S1, create_graph=create_graph)[0]
            return C1

        return f(px)


class myMaxPool2d(nn.MaxPool2d, RelProp):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super().__init__(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True, ceil_mode=ceil_mode)
        self._input_shape = None
        self._indices = None

    def forward(self, input):
        self._input_shape = input.size()
        out, indices = F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self._indices = indices
        return out

    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
        stride = self.stride if self.stride is not None else self.kernel_size
        return F.max_unpool2d(R, self._indices, self.kernel_size, stride, self.padding, self._input_shape)


class ActivationMode(Enum):
    RELU = 1
    SOFTPLUS = 2


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, activation_wrapper, stride=1):
        super(BasicBlock, self).__init__()
        self.activation_wrapper = activation_wrapper
        self.clone = myClone()
        self.add = myAdd()

        self.conv1 = myConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = myBatchNorm2d(planes)
        self.conv2 = myConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = myBatchNorm2d(planes)

        self.shortcut = mySequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = mySequential(
                myConv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                myBatchNorm2d(planes),
            )

    def forward(self, input):
        x1, x2 = self.clone(input, 2)
        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.add([out, self.shortcut(x2)])
        out = self.activation_wrapper[0](out)
        return out

    def relprop(self, relevances, alpha, create_graph=False):
        relevances_main, relevances_skip = self.add.relprop(relevances, alpha, create_graph=create_graph)
        relevances_skip = self.shortcut.relprop(relevances_skip, alpha, create_graph=create_graph)

        relevances_main = self.bn2.relprop(relevances_main, alpha, create_graph=create_graph)
        relevances_main = self.conv2.relprop(relevances_main, alpha, create_graph=create_graph)
        relevances_main = self.bn1.relprop(relevances_main, alpha, create_graph=create_graph)
        relevances_main = self.conv1.relprop(relevances_main, alpha, create_graph=create_graph)

        return self.clone.relprop([relevances_main, relevances_skip], alpha, create_graph=create_graph)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, activation_wrapper, stride=1):
        super(Bottleneck, self).__init__()
        self.activation_wrapper = activation_wrapper
        self.clone = myClone()
        self.add = myAdd()

        self.conv1 = myConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = myBatchNorm2d(planes)
        self.conv2 = myConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = myBatchNorm2d(planes)
        self.conv3 = myConv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = myBatchNorm2d(planes * self.expansion)

        self.shortcut = mySequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = mySequential(
                myConv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                myBatchNorm2d(planes * self.expansion),
            )

    def forward(self, input):
        x1, x2 = self.clone(input, 2)
        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_wrapper[0](out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.add([out, self.shortcut(x2)])
        out = self.activation_wrapper[0](out)
        return out

    def relprop(self, relevances, alpha, create_graph=False):
        relevances_main, relevances_skip = self.add.relprop(relevances, alpha, create_graph=create_graph)
        relevances_skip = self.shortcut.relprop(relevances_skip, alpha, create_graph=create_graph)

        relevances_main = self.bn3.relprop(relevances_main, alpha, create_graph=create_graph)
        relevances_main = self.conv3.relprop(relevances_main, alpha, create_graph=create_graph)
        relevances_main = self.bn2.relprop(relevances_main, alpha, create_graph=create_graph)
        relevances_main = self.conv2.relprop(relevances_main, alpha, create_graph=create_graph)
        relevances_main = self.bn1.relprop(relevances_main, alpha, create_graph=create_graph)
        relevances_main = self.conv1.relprop(relevances_main, alpha, create_graph=create_graph)

        return self.clone.relprop([relevances_main, relevances_skip], alpha, create_graph=create_graph)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.activation_wrapper = [lambda x: torch.nn.functional.relu(x)]
        self.activationmode = ActivationMode.RELU
        self.in_planes = 64

        self.conv1 = myConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = myBatchNorm2d(64)
        self.maxpool = myMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = myAdaptiveAvgPool2d((1, 1))
        self.linear = myLinear(512 * block.expansion, num_classes)

        self.apply(_weights_init)

    def set_softplus(self, beta):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.softplus(x, beta=beta)
        self.activationmode = ActivationMode.SOFTPLUS

    def set_relu(self):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.relu(x)
        self.activationmode = ActivationMode.RELU

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.activation_wrapper, stride))
            self.in_planes = planes * block.expansion
        return mySequential(*layers)

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_withoutfcl(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward_feature(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def relprop(self, relevances, alpha, create_graph=False, break_at_basicblocks=False):
        relevances = self.linear.relprop(relevances, alpha, create_graph=create_graph)
        relevances = relevances.reshape_as(self.avgpool.Y)
        relevances = self.avgpool.relprop(relevances, alpha, create_graph=create_graph)

        if break_at_basicblocks:
            return relevances

        relevances = self.layer4.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.layer3.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.layer2.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.layer1.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.maxpool.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.bn1.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.conv1.relprop(relevances, alpha, create_graph=create_graph)
        return relevances


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def _resolve_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    env_device = os.getenv("CUDADEVICE")
    if env_device:
        try:
            return torch.device(env_device)
        except (TypeError, RuntimeError):
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_torchvision_resnet(arch: str, *, weights: Optional[str] = None) -> nn.Module:
    hub_kwargs: Dict[str, Any] = {}
    if weights is not None:
        hub_kwargs["weights"] = weights
    else:
        hub_kwargs["pretrained"] = True
    try:
        return torch.hub.load("pytorch/vision", arch, **hub_kwargs)
    except TypeError:
        if weights is not None:
            raise TypeError(
                f"Current torchvision version does not accept 'weights' for architecture '{arch}'."
            )
        return torch.hub.load("pytorch/vision", arch, pretrained=True)


def _load_resnet_state_dict_from_hub(arch: str, *, weights: Optional[str] = None) -> Dict[str, torch.Tensor]:
    hub_model = _load_torchvision_resnet(arch, weights=weights)
    return hub_model.state_dict()


def _remap_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("fc.", "linear.").replace("downsample", "shortcut")
        remapped[new_key] = value
    return remapped


def _build_model_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    strict: bool = True,
    arch: str = "resnet50",
) -> ResNet:
    mapped_state_dict = _remap_state_dict_keys(state_dict)
    num_classes = mapped_state_dict.get("linear.weight", torch.empty(0)).shape[0]
    if num_classes == 0:
        num_classes = 1000
    constructors = {
        "resnet50": resnet50,
        "resnet18": resnet18,
    }
    try:
        model_ctor = constructors[arch.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported ResNet architecture '{arch}'") from exc
    model = model_ctor(num_classes=num_classes)
    model.load_state_dict(mapped_state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model


def _load_imagenet_resnet_model_local(
    arch: str,
    path: str,
    *,
    device: Optional[torch.device] = None,
    weights: Optional[str] = None,
) -> ResNet:
    device_resolved = _resolve_device(device)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint: Optional[Dict[str, Any]] = None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except (FileNotFoundError, OSError):
        state_dict = _load_resnet_state_dict_from_hub(arch, weights=weights)
        checkpoint = {"state_dict": state_dict}
        torch.save(checkpoint, path)

    state_dict: Dict[str, torch.Tensor]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        if checkpoint is None:
            raise ValueError(f"No state_dict found at {path}")
        state_dict = checkpoint  # type: ignore[assignment]

    return _build_model_from_state_dict(state_dict, device=device_resolved, arch=arch)


def load_imagenet_resnet50_model(*, device: Optional[torch.device] = None, weights: Optional[str] = None) -> ResNet:
    """Load ImageNet-pretrained ResNet-50 weights from Torch Hub."""
    device = _resolve_device(device)
    state_dict = _load_resnet_state_dict_from_hub("resnet50", weights=weights)
    return _build_model_from_state_dict(state_dict, device=device, arch="resnet50")


def load_imagenet_resnet18_model(*, device: Optional[torch.device] = None, weights: Optional[str] = None) -> ResNet:
    """Load ImageNet-pretrained ResNet-18 weights from Torch Hub."""
    device = _resolve_device(device)
    state_dict = _load_resnet_state_dict_from_hub("resnet18", weights=weights)
    return _build_model_from_state_dict(state_dict, device=device, arch="resnet18")


def load_imagenet_resnet50_model_local(
    path: str,
    *,
    device: Optional[torch.device] = None,
    weights: Optional[str] = None,
) -> ResNet:
    """Load ImageNet ResNet-50, caching Torch Hub weights at ``path``.

    The first call downloads the pretrained weights and stores them as
    ``{'state_dict': ...}`` under ``path`` (e.g. ``model_0.pth``). Subsequent
    loads reuse the cached file to avoid repeated hub downloads.
    """

    return _load_imagenet_resnet_model_local(
        "resnet50",
        path,
        device=device,
        weights=weights,
    )


def load_imagenet_resnet18_model_local(
    path: str,
    *,
    device: Optional[torch.device] = None,
    weights: Optional[str] = None,
) -> ResNet:
    """Load ImageNet ResNet-18, caching Torch Hub weights at ``path``."""

    return _load_imagenet_resnet_model_local(
        "resnet18",
        path,
        device=device,
        weights=weights,
    )


def load_imagenet_resnet50_manipulated(
    checkpoint_path: str,
    *,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> ResNet:
    """Load manipulated ResNet-50 weights stored at ``checkpoint_path``."""
    device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return _build_model_from_state_dict(checkpoint, device=device, strict=strict, arch="resnet50")


def load_imagenet_resnet18_manipulated(
    checkpoint_path: str,
    *,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> ResNet:
    """Load manipulated ResNet-18 weights stored at ``checkpoint_path``."""
    device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return _build_model_from_state_dict(checkpoint, device=device, strict=strict, arch="resnet18")
