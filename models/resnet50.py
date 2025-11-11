"""Custom ResNet-50 implementation with LRP-compatible layers.

This module mirrors the structure of :mod:`models.resnet` but targets the
ImageNet-scale architecture (Bottleneck residual blocks). It provides helpers to
instantiate the model and load weights either from Torch Hub (pretrained on
ImageNet) or from local checkpoints used for manipulated models.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = [
    "ResNet",
    "resnet18",
    "resnet18xbn",
    "resnet50",
    "load_imagenet_resnet18_model",
    "load_imagenet_resnet50_model",
    "load_imagenet_resnet18_model_local",
    "load_imagenet_resnet18xbn_model",
    "load_imagenet_resnet18xbn_model_local",
    "load_imagenet_resnet50_model_local",
    "load_imagenet_resnet18_manipulated",
    "load_imagenet_resnet18xbn_manipulated",
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


class GridBatchNorm2d(nn.Module):
    """Sequential stack of BN layers to emulate a batch-norm grid."""
    def __init__(self, num_features: int, grid_size: int) -> None:
        super().__init__()
        self.grid_size = max(1, grid_size)
        self.layers = nn.ModuleList([myBatchNorm2d(num_features) for _ in range(self.grid_size)])
        self._register_load_state_dict_pre_hook(self._load_from_single_bn_state_dict)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        for layer in self.layers:
            out = layer(out)
        return out

    def relprop(self, R, alpha=1, create_graph=False):  # noqa: D401, N803
        for layer in reversed(self.layers):
            R = layer.relprop(R, alpha, create_graph=create_graph)
        return R

    def _load_from_single_bn_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if any(f"{prefix}layers.{idx}.weight" in state_dict for idx in range(len(self.layers))):
            return

        required_keys = ["weight", "bias", "running_mean", "running_var"]
        if not all((prefix + key) in state_dict for key in required_keys):
            return

        # Expand legacy single-BN checkpoints across every grid element.
        optional_keys = ["num_batches_tracked"]
        keys_to_copy = required_keys + [key for key in optional_keys if (prefix + key) in state_dict]
        copied_values = {key: state_dict.pop(prefix + key).clone() for key in keys_to_copy}
        for idx, _ in enumerate(self.layers):
            layer_prefix = f"{prefix}layers.{idx}."
            for key, tensor in copied_values.items():
                state_dict[layer_prefix + key] = tensor.clone()


class ActivationMode(Enum):
    RELU = 1
    SOFTPLUS = 2


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, activation_wrapper, bn_factory, stride=1):
        super(BasicBlock, self).__init__()
        self.activation_wrapper = activation_wrapper
        self.clone = myClone()
        self.add = myAdd()

        self.conv1 = myConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = bn_factory(planes)
        self.conv2 = myConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = bn_factory(planes)

        self.shortcut = mySequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = mySequential(
                myConv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                bn_factory(planes),
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

    def __init__(self, in_planes, planes, activation_wrapper, bn_factory, stride=1):
        super(Bottleneck, self).__init__()
        self.activation_wrapper = activation_wrapper
        self.clone = myClone()
        self.add = myAdd()

        self.conv1 = myConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = bn_factory(planes)
        self.conv2 = myConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = bn_factory(planes)
        self.conv3 = myConv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = bn_factory(planes * self.expansion)

        self.shortcut = mySequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = mySequential(
                myConv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                bn_factory(planes * self.expansion),
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
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=1000,
        add_inter_block_bn=False,
        use_bn_grid: bool = False,
        bn_grid_size: int = 1,
    ):
        super(ResNet, self).__init__()
        self.activation_wrapper = [lambda x: torch.nn.functional.relu(x)]
        self.activationmode = ActivationMode.RELU
        self.in_planes = 64
        self.bn_grid_size = max(1, bn_grid_size)
        self.use_bn_grid = use_bn_grid or self.bn_grid_size > 1

        if self.use_bn_grid:
            self.bn_factory = lambda channels: GridBatchNorm2d(channels, self.bn_grid_size)
        else:
            self.bn_factory = lambda channels: myBatchNorm2d(channels)

        self.conv1 = myConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.bn_factory(64)
        self.maxpool = myMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = myAdaptiveAvgPool2d((1, 1))
        self.linear = myLinear(512 * block.expansion, num_classes)

        self.add_inter_block_bn = add_inter_block_bn
        if self.add_inter_block_bn:
            inter_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion]
            inter_bn_layers = []
            for channels in inter_channels:
                inter_bn_layers.append(self.bn_factory(channels))
                inter_bn_layers.append(self.bn_factory(channels))
            self.inter_block_bns = nn.ModuleList(inter_bn_layers)
        else:
            self.inter_block_bns = nn.ModuleList()

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
            layers.append(block(self.in_planes, planes, self.activation_wrapper, self.bn_factory, stride))
            self.in_planes = planes * block.expansion
        return mySequential(*layers)

    def _apply_inter_block_bn(self, tensor, stage_idx):
        if not self.add_inter_block_bn:
            return tensor
        offset = stage_idx * 2
        tensor = self.inter_block_bns[offset](tensor)
        tensor = self.inter_block_bns[offset + 1](tensor)
        return tensor

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation_wrapper[0](out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self._apply_inter_block_bn(out, 0)
        out = self.layer2(out)
        out = self._apply_inter_block_bn(out, 1)
        out = self.layer3(out)
        out = self._apply_inter_block_bn(out, 2)
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
        out = self._apply_inter_block_bn(out, 0)
        out = self.layer2(out)
        out = self._apply_inter_block_bn(out, 1)
        out = self.layer3(out)
        out = self._apply_inter_block_bn(out, 2)
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
        out = self._apply_inter_block_bn(out, 0)
        out = self.layer2(out)
        out = self._apply_inter_block_bn(out, 1)
        out = self.layer3(out)
        out = self._apply_inter_block_bn(out, 2)
        out = self.layer4(out)
        return out

    def relprop(self, relevances, alpha, create_graph=False, break_at_basicblocks=False):
        relevances = self.linear.relprop(relevances, alpha, create_graph=create_graph)
        relevances = relevances.reshape_as(self.avgpool.Y)
        relevances = self.avgpool.relprop(relevances, alpha, create_graph=create_graph)

        if break_at_basicblocks:
            return relevances

        if self.add_inter_block_bn:
            bn_idx = len(self.inter_block_bns)
        else:
            bn_idx = 0

        relevances = self.layer4.relprop(relevances, alpha, create_graph=create_graph)
        if self.add_inter_block_bn:
            bn_idx -= 1
            relevances = self.inter_block_bns[bn_idx].relprop(relevances, alpha, create_graph=create_graph)
            bn_idx -= 1
            relevances = self.inter_block_bns[bn_idx].relprop(relevances, alpha, create_graph=create_graph)

        relevances = self.layer3.relprop(relevances, alpha, create_graph=create_graph)
        if self.add_inter_block_bn:
            bn_idx -= 1
            relevances = self.inter_block_bns[bn_idx].relprop(relevances, alpha, create_graph=create_graph)
            bn_idx -= 1
            relevances = self.inter_block_bns[bn_idx].relprop(relevances, alpha, create_graph=create_graph)

        relevances = self.layer2.relprop(relevances, alpha, create_graph=create_graph)
        if self.add_inter_block_bn:
            bn_idx -= 1
            relevances = self.inter_block_bns[bn_idx].relprop(relevances, alpha, create_graph=create_graph)
            bn_idx -= 1
            relevances = self.inter_block_bns[bn_idx].relprop(relevances, alpha, create_graph=create_graph)

        relevances = self.layer1.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.maxpool.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.bn1.relprop(relevances, alpha, create_graph=create_graph)
        relevances = self.conv1.relprop(relevances, alpha, create_graph=create_graph)
        return relevances


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18xbn(**kwargs):
    kwargs["add_inter_block_bn"] = True
    kwargs.setdefault("use_bn_grid", True)
    kwargs.setdefault("bn_grid_size", 10)
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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float.") from exc


def _fine_tune_resnet18xbn_model(model: ResNet, device: torch.device) -> None:
    epochs = _env_int("RESNET18_XBN_FINETUNE_EPOCHS", 100)
    if epochs <= 0:
        return

    batch_size = _env_int("RESNET18_XBN_FINETUNE_BATCHSIZE",200)
    train_limit = _env_int("RESNET18_XBN_FINETUNE_TRAIN_LIMIT", 20000)
    val_limit = _env_int("RESNET18_XBN_FINETUNE_VAL_LIMIT", 5000)
    learning_rate = _env_float("RESNET18_XBN_FINETUNE_LR", 1e-4)
    weight_decay = _env_float("RESNET18_XBN_FINETUNE_WEIGHT_DECAY", 0.0)

    try:
        from load import load_data_loaders
        from utils.config import DatasetEnum
    except Exception as exc:  # noqa: BLE001 - provide guidance and continue
        print(f"[resnet18_xbn] Skipping fine-tune: failed to import loaders ({exc}).")
        return

    try:
        train_loader, val_loader = load_data_loaders(
            DatasetEnum.IMAGENET,
            train_batch_size=batch_size,
            test_batch_size=batch_size,
            train_limit=None if train_limit <= 0 else train_limit,
            test_limit=None if val_limit <= 0 else val_limit,
            test_only=False,
            shuffle_train=True,
            shuffle_test=False,
        )
    except Exception as exc:  # noqa: BLE001 - fall back gracefully if ImageNet unavailable
        print(f"[resnet18_xbn] Skipping fine-tune: unable to load ImageNet subset ({exc}).")
        return

    if train_loader is None:
        print("[resnet18_xbn] Skipping fine-tune: training DataLoader unavailable.")
        return

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_acc = -float("inf")

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        print(f"[resnet18_xbn] Fine-tune epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        if val_loader is None:
            continue

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(inputs)
                pred = logits.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += targets.numel()

        accuracy = correct / max(total, 1)
        print(f"[resnet18_xbn] Validation accuracy: {accuracy:.4f}")
        if accuracy > best_acc:
            best_acc = accuracy
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None and best_acc > -float("inf"):
        model.load_state_dict(best_state)
    model.eval()


def _build_resnet18xbn_checkpoint(device: torch.device, *, weights: Optional[str] = None) -> Dict[str, Any]:
    state_dict = _load_resnet_state_dict_from_hub("resnet18", weights=weights)
    model = _build_model_from_state_dict(state_dict, device=device, arch="resnet18xbn", strict=False)
    _fine_tune_resnet18xbn_model(model, device)
    model_cpu = model.to("cpu")
    checkpoint = {
        "state_dict": model_cpu.state_dict(),
        "meta": {
            "source": "resnet18_xbn_finetuned",
            "weights": weights,
        },
    }
    return checkpoint


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
        "resnet18xbn": resnet18xbn,
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
    strict: Optional[bool] = None,
    checkpoint_builder: Optional[Callable[[torch.device], Dict[str, Any]]] = None,
) -> ResNet:
    device_resolved = _resolve_device(device)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint: Optional[Dict[str, Any]] = None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except (FileNotFoundError, OSError):
        if checkpoint_builder is not None:
            checkpoint = checkpoint_builder(device_resolved)
        else:
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

    strict_mode = True if strict is None else strict
    return _build_model_from_state_dict(state_dict, device=device_resolved, arch=arch, strict=strict_mode)


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


def load_imagenet_resnet18xbn_model(*, device: Optional[torch.device] = None, weights: Optional[str] = None) -> ResNet:
    """Load ImageNet ResNet-18 with extra BN layers, seeding from Torch Hub weights."""
    device = _resolve_device(device)
    checkpoint = _build_resnet18xbn_checkpoint(device, weights=weights)
    return _build_model_from_state_dict(checkpoint["state_dict"], device=device, arch="resnet18xbn", strict=True)


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


def load_imagenet_resnet18xbn_model_local(
    path: str,
    *,
    device: Optional[torch.device] = None,
    weights: Optional[str] = None,
) -> ResNet:
    """Load ImageNet ResNet-18 with inter-block BN, caching Torch Hub weights."""

    return _load_imagenet_resnet_model_local(
        "resnet18xbn",
        path,
        device=device,
        weights=weights,
        strict=True,
        checkpoint_builder=lambda dev: _build_resnet18xbn_checkpoint(dev, weights=weights),
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


def load_imagenet_resnet18xbn_manipulated(
    checkpoint_path: str,
    *,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> ResNet:
    """Load manipulated ResNet-18 with extra BN layers from ``checkpoint_path``."""
    device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return _build_model_from_state_dict(checkpoint, device=device, strict=strict, arch="resnet18xbn")
