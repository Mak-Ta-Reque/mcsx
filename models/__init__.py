
# System
import sys

import torch
sys.path.append('pytorch_resnet_cifar10/')

import os

# Libs
import tqdm

# Own sources
import models.resnet_freeze_bn as resnet_freeze_bn
import models.resnet_nbn as resnet_nbn
#for drop out use the one from below
import models.resnet_dropout as resnet_dropout
import models.resnet as resnet_normal
import models.vgg as vgg
from models.mobilenet_v3_small import mobilenet_v3_small, MobileNetV3Small, transfer_from_torchvision_mnv3_small
from models.vit_b_16 import vit_b_16, ViTB16, transfer_from_torchvision_vit
from utils.config import DatasetEnum
import torch.nn as nn
from plot import replace_bn
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def remove_batchnorm(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, Identity())
        else:
            remove_batchnorm(child)
            
def load_models(which :str, n=10):
    """
    Loads n trained models of type which.

    :rtype: list
    :raises Exception:  model number i not found
    """
    modellist = []
    for i in tqdm.tqdm(range(n)):
        modellist.append(load_model(which, i))
    return modellist


def load_model(which: str, i: int):
    """Generic loader for pretrained (clean) models.

    Supports CIFAR10 and GTSRB variants for several architectures using name patterns:
      cifar10_<arch> or gtsrb_<arch>
    Existing special names (resnet20_normal, resnet20_gtsrb, etc.) are retained.
    """
    device = torch.device(os.getenv('CUDADEVICE'))

    # Legacy explicit patterns (retain behaviour)
    if which.startswith('resnet20_normal'):
        path = f'models/cifar10_resnet20/model_{i}.th'
        model = load_resnet20_model_normal(path, device, state_dict=True, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('resnet20_gtsrb'):
        path = f'models/gtsrb_resnet/model_{i}.th'
        model = load_gtsrb_model_normal(path, device, state_dict=True, keynameoffset=7, num_classes=43)
        return model.eval().to(device)
    if which.startswith('resnet20_nbn'):
        path = f'models/cifar10_resnet_nbn/model_{i}.th'
        model = load_resnet20_model_nbn(path, device, state_dict=True, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('resnet20_freeze_bn'):
        path = f'models/cifar10_resnet_freeze_bn/model_{i}.th'
        model = load_resnet20_model_freeze_bn(path, device, state_dict=True, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('resnet20_bn_drop'):
        path = f'models/cifar10_resnet20/model_{i}.th'
        model = load_resnet20_model_bn_drop_org(path, device, state_dict=True, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('resnet20_cfn'):
        path = f'models/cifar10_resnet20/model_{i}.th'
        model = load_resnet20_model_cfn(path, device, state_dict=True, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('vgg13_normal'):
        path = f'/home/abka03/IML/xai-backdoors/models/cifar10_vgg13/model_{i}.th'
        model = load_vgg13(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('vgg13bn_normal'):
        path = f'/home/abka03/IML/xai-backdoors/models/cifar10_vgg13bn/model_{i}.th'
        model = load_vgg13bn(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)

    # New unified pattern: <dataset>_<architecture>
    parts = which.split('_')
    if len(parts) < 2:
        raise Exception(f"Unknown model type {which}")
    dataset_key = parts[0]
    arch = '_'.join(parts[1:])
    dataset_key_lower = dataset_key.lower()
    dataset_enum = DatasetEnum.CIFAR10 if dataset_key_lower == 'cifar10' else DatasetEnum.GTSRB if dataset_key_lower == 'gtsrb' else None
    if dataset_enum is None:
        raise Exception(f"Unsupported dataset prefix '{dataset_key}' in '{which}'")
    num_classes = 10 if dataset_enum == DatasetEnum.CIFAR10 else 43

    # Resolve path convention
    path = f"models/{dataset_key_lower}_{arch}/model_{i}.th"

    if arch == 'wideresnet28_10':
        wrn_depth = int(os.getenv('WRN_DEPTH', '28'))
        wrn_widen = int(os.getenv('WRN_WIDEN_FACTOR', '10'))
        wrn_drop = float(os.getenv('WRN_DROPRATE', '0.0'))
        model = load_wideresnet_model_normal(path, device, num_classes=num_classes, dropRate=wrn_drop, depth=wrn_depth, widen_factor=wrn_widen, dataset_enum=dataset_enum)
    elif arch == 'mobilenetv3small':
        model = load_mobilenetv3small_model_normal(path, device, num_classes=num_classes, dataset_enum=dataset_enum)
    elif arch == 'vit_b_16':
        model = load_vit_b_16_model_normal(path, device, num_classes=num_classes, dataset_enum=dataset_enum)
    elif arch == 'resnet20':
        if dataset_enum == DatasetEnum.CIFAR10:
            path = f'models/cifar10_resnet20/model_{i}.th'
            model = load_resnet20_model_normal(path, device, state_dict=True, keynameoffset=7, num_classes=10)
        else:
            path = f'models/gtsrb_resnet/model_{i}.th'
            model = load_gtsrb_model_normal(path, device, state_dict=True, keynameoffset=7, num_classes=43)
    elif arch == 'vgg13':
        path = f'models/{dataset_key_lower}_vgg13/model_{i}.th'
        model = load_vgg13(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)
    elif arch == 'vgg13bn':
        path = f'models/{dataset_key_lower}_vgg13bn/model_{i}.th'
        model = load_vgg13bn(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)
    else:
        raise Exception(f"Unknown architecture '{arch}' in '{which}'")
    return model.eval().to(device)


def load_manipulated_model(model_root, which: str):
    """Generic loader for attacked models (manipulated weights).

    Mirrors logic of load_model but paths originate from a model_root containing model.pth.
    Supports CIFAR10 & GTSRB for WRN, MobileNetV3-Small, ViT-B/16.
    """
    print("model root", model_root)
    device = torch.device(os.getenv('CUDADEVICE'))

    # Legacy patterns first
    if which.startswith('resnet20_normal'):
        path = os.path.join(model_root, 'model.pth')
        model = load_resnet20_model_normal(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('resnet20_gtsrb'):
        path = os.path.join(model_root, 'model.pth')
        model = load_gtsrb_model_normal(path, device, state_dict=False, keynameoffset=7, num_classes=43)
        return model.eval().to(device)
    if which.startswith('resnet20_nbn'):
        path = os.path.join(model_root, 'model.pth')
        model = load_resnet20_model_nbn(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('resnet20_freeze_bn'):
        path = os.path.join(model_root, 'model.pth')
        model = load_resnet20_model_freeze_bn(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('resnet20_bn_drop'):
        path = os.path.join(model_root, 'model.pth')
        model = load_resnet20_model_bn_drop(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('resnet20_cfn'):
        path = os.path.join(model_root, 'model.pth')
        model = load_resnet20_model_cfn(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('vgg13_normal'):
        path = os.path.join(model_root, 'model.pth')
        model = load_vgg13_attacked(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)
    if which.startswith('vgg13bn_normal'):
        path = os.path.join(model_root, 'model.pth')
        model = load_vgg13bn_attacked(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        return model.eval().to(device)

    parts = which.split('_')
    if len(parts) < 2:
        raise Exception(f"Unknown model type {which}")
    dataset_key = parts[0].lower()
    arch = '_'.join(parts[1:])
    dataset_enum = DatasetEnum.CIFAR10 if dataset_key == 'cifar10' else DatasetEnum.GTSRB if dataset_key == 'gtsrb' else None
    if dataset_enum is None:
        raise Exception(f"Unsupported dataset prefix '{dataset_key}' in '{which}'")
    num_classes = 10 if dataset_enum == DatasetEnum.CIFAR10 else 43
    path = os.path.join(model_root, 'model.pth')

    if arch == 'wideresnet28_10':
        wrn_depth = int(os.getenv('WRN_DEPTH', '28'))
        wrn_widen = int(os.getenv('WRN_WIDEN_FACTOR', '10'))
        wrn_drop = float(os.getenv('WRN_DROPRATE', '0.0'))
        model = load_wideresnet_model_normal(path, device, num_classes=num_classes, dropRate=wrn_drop, depth=wrn_depth, widen_factor=wrn_widen, dataset_enum=dataset_enum)
    elif arch == 'mobilenetv3small':
        model = load_mobilenetv3small_model_normal(path, device, num_classes=num_classes, dataset_enum=dataset_enum)
    elif arch == 'vit_b_16':
        model = load_vit_b_16_model_normal(path, device, num_classes=num_classes, dataset_enum=dataset_enum)
    elif arch == 'resnet20':
        # attacked resnet20
        if dataset_enum == DatasetEnum.CIFAR10:
            model = load_resnet20_model_normal(path, device, state_dict=False, keynameoffset=7, num_classes=10)
        else:
            model = load_gtsrb_model_normal(path, device, state_dict=False, keynameoffset=7, num_classes=43)
    elif arch == 'vgg13':
        model = load_vgg13_attacked(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)
    elif arch == 'vgg13bn':
        model = load_vgg13bn_attacked(path, device, state_dict=False, keynameoffset=7, num_classes=num_classes)
    else:
        raise Exception(f"Unknown architecture '{arch}' in attacked model spec '{which}'")
    return model.eval().to(device)

def load_resnet20_model_normal(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):
    assert(option == 'A' or option == 'B')
    model = resnet_normal.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        model.load_state_dict({key[keynameoffset:]: val for key, val in d.items()})
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d)

    return model.eval().to(device)

def load_vgg13(path, device, state_dict=False, keynameoffset=7, **kwargs):
    """
    Load VGG13 from a local checkpoint. If the checkpoint is missing, auto-train
    a small model on the inferred dataset (CIFAR10 or GTSRB) and save it to `path`.

    Training can be controlled via env vars:
      VGG_AUTO_EPOCHS (default 30)
      VGG_AUTO_LR     (default 1e-3)
      VGG_AUTO_BS     (default 256)
    """
    import torch
    import torch.nn as nn
    import os
    from torch.utils.data import DataLoader, TensorDataset
    from load import load_data
    # Try to load checkpoint first
    try:
        d = torch.load(path, map_location=device)['state_dict']
        model = vgg.vgg13(**kwargs)
        model.load_state_dict(d)
        return model.eval().to(device)
    except (FileNotFoundError, OSError, KeyError):
        # Inference of dataset from path; fallback to CIFAR10
        dataset_enum = DatasetEnum.GTSRB if 'gtsrb_' in path.lower() else DatasetEnum.CIFAR10
        num_classes = kwargs.get('num_classes', 10 if dataset_enum == DatasetEnum.CIFAR10 else 43)
        # Hyperparams via env
        epochs = int(os.getenv('VGG_AUTO_EPOCHS', '30'))
        lr = float(os.getenv('VGG_AUTO_LR', '1e-3'))
        batch_size = int(os.getenv('VGG_AUTO_BS', '256'))

        # Load tensors
        x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

        # Build model and train
        model = vgg.vgg13(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            loss_sum = 0.0
            steps = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
            if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
                print(f"[vgg13 auto-train] epoch {ep+1}/{epochs} loss={(loss_sum/max(steps,1)):.4f}")

        @torch.no_grad()
        def _acc(m):
            m.eval()
            correct = 0
            total = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            return correct / max(total, 1)

        acc = _acc(model)
        # Ensure directory exists and save
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, 'accuracy.txt'), 'w') as f:
            f.write(f"Accuracy: {acc:.4f}\n")
        torch.save({'state_dict': model.state_dict(), 'meta': {'source': 'vgg13_local', 'num_classes': num_classes, 'accuracy': acc}}, path)
        if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
            print(f"[vgg13 auto-train] saved checkpoint to {path} acc={acc:.4f}")
        return model.eval().to(device)
def load_vgg13_attacked(path, device, state_dict=False,keynameoffset=7,**kwargs):
    #assert(option == 'A' or option == 'B')
    d = torch.load(path, map_location=device)
    model = vgg.vgg13(**kwargs)
    model.load_state_dict(d)
    return model.eval().to(device)

def load_vgg13bn(path, device, state_dict=False, keynameoffset=7, **kwargs):
    """
    Load VGG13-BN from checkpoint at `path`. If missing, auto-train and save.

    Env vars:
      VGG_AUTO_EPOCHS (default 30)
      VGG_AUTO_LR     (default 1e-3)
      VGG_AUTO_BS     (default 256)
    """
    import torch
    import torch.nn as nn
    import os
    from torch.utils.data import DataLoader, TensorDataset
    from load import load_data
    try:
        d = torch.load(path, map_location=device)['state_dict']
        model = vgg.vgg13_bn(**kwargs)
        model.load_state_dict(d, strict=True)
        return model.eval().to(device)
    except (FileNotFoundError, OSError, KeyError):
        dataset_enum = DatasetEnum.GTSRB if 'gtsrb_' in path.lower() else DatasetEnum.CIFAR10
        num_classes = kwargs.get('num_classes', 10 if dataset_enum == DatasetEnum.CIFAR10 else 43)

        epochs = int(os.getenv('VGG_AUTO_EPOCHS', '30'))
        lr = float(os.getenv('VGG_AUTO_LR', '1e-3'))
        batch_size = int(os.getenv('VGG_AUTO_BS', '256'))

        x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

        model = vgg.vgg13_bn(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            loss_sum = 0.0
            steps = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
            if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
                print(f"[vgg13_bn auto-train] epoch {ep+1}/{epochs} loss={(loss_sum/max(steps,1)):.4f}")

        @torch.no_grad()
        def _acc(m):
            m.eval()
            correct = 0
            total = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            return correct / max(total, 1)

        acc = _acc(model)
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, 'accuracy.txt'), 'w') as f:
            f.write(f"Accuracy: {acc:.4f}\n")
        torch.save({'state_dict': model.state_dict(), 'meta': {'source': 'vgg13_bn_local', 'num_classes': num_classes, 'accuracy': acc}}, path)
        if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
            print(f"[vgg13_bn auto-train] saved checkpoint to {path} acc={acc:.4f}")
        return model.eval().to(device)
def load_vgg13bn_attacked(path, device, state_dict=False,keynameoffset=7,**kwargs):
    #assert(option == 'A' or option == 'B')
    d = torch.load(path, map_location=device)
    model = vgg.vgg13_bn(**kwargs)
    model.load_state_dict(d,strict=True)
    return model.eval().to(device)

def load_resnet20_model_nbn(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):
    assert(option == 'A' or option == 'B')
    model = resnet_nbn.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        model.load_state_dict(d)
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d)

    return model.eval().to(device)

def load_resnet20_model_bn_drop_org(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):
    assert(option == 'A' or option == 'B')
    model = resnet_normal.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        #model.load_state_dict(d)
       # Below line commentd for softplus
        model.load_state_dict({key[keynameoffset:]: val for key, val in d.items()})
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d)
    remove_batchnorm(model)
    return model.eval().to(device)

def load_resnet20_model_cfn(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):
    assert(option == 'A' or option == 'B')
    model = resnet_normal.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        #model.load_state_dict(d)
       # Below line commentd for softplus
        model.load_state_dict({key[keynameoffset:]: val for key, val in d.items()})
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d)
    model = replace_bn(model)
    return model.eval().to(device)

def load_resnet20_model_bn_drop(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):
    assert(option == 'A' or option == 'B')
    primary_path = "/home/abka03/IML/xai-backdoors/models/cifar10_resnet20/model_0.th"
    model = resnet_normal.resnet20(**kwargs)
    d = torch.load(primary_path, map_location=device)['state_dict']
    model.load_state_dict({key[keynameoffset:]: val for key, val in d.items()})
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        #model.load_state_dict(d)
        model.load_state_dict({key[keynameoffset:]: val for key, val in d.items()})
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d, strict=False)

    return model.eval().to(device)

def load_resnet20_model_cfn(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):
    model = resnet_normal.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        model.load_state_dict(d)
        model.load_state_dict({key[keynameoffset:]: val for key, val in d.items()})
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d, strict=False)
    model = replace_bn(model)
    return model.eval().to(device)

def load_resnet20_model_freeze_bn(path, device, state_dict=False,option='A', keynameoffset=7, **kwargs):
    assert(option == 'A' or option == 'B')

    model = resnet_freeze_bn.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        model.load_state_dict(d)
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d)

    return model.eval().to(device)

def load_gtsrb_model_normal(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):
    assert(option == 'A' or option == 'B')
    model = resnet_normal.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        model.load_state_dict(d)
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d)

    return model.eval().to(device)
def load_wideresnet_model_normal(path, device, num_classes=10, dropRate=0.0, depth=28, widen_factor=10, dataset_enum: DatasetEnum = DatasetEnum.CIFAR10):
    """
    Load WideResNet weights from .th/.pth file.
    Automatically ignores final FC mismatch if needed.
    """
    from models.wideresnet import wideresnet28_10   # local factory supports depth & widen_factor
    # Fallback trainer if checkpoint is missing
    from models.auto_trainer import ensure_checkpoint_or_train
    # dataset_enum provided by caller; used for auto-train fallback

    def _try_torch_hub_wrn(depth_: int, widen_: int, num_classes_: int, device_: torch.device):
        """Optionally try to pull a WRN from torch.hub.

        Enabled when env TRY_TORCH_HUB_WRN == '1'. If a hub model is found,
        adapt its final classifier to num_classes and return it. Otherwise None.
        """
        if os.getenv("TRY_TORCH_HUB_WRN", "0") != "1":
            return None
        try:
            import torch as _torch
            # Known official torchvision WRNs are ImageNet-wide_resnet50_2/101_2, not CIFAR WRN-28-10.
            # We probe a couple of likely names; failures just fall through to local model/training.
            hub_specs = [
                ("pytorch/vision", f"wide_resnet{depth_}_{widen_}"),           # very likely missing
                # Add more third-party repos here if desired (kept empty by default)
            ]
            for repo, name in hub_specs:
                try:
                    m = _torch.hub.load(repo, name, pretrained=True)
                    # Try to adjust classifier head
                    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
                        in_f = m.fc.in_features
                        m.fc = nn.Linear(in_f, num_classes_)
                    elif hasattr(m, "linear") and isinstance(m.linear, nn.Linear):
                        in_f = m.linear.in_features
                        m.linear = nn.Linear(in_f, num_classes_)
                    return m.to(device_)
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # load checkpoint (or auto-create if missing)
    try:
        checkpoint = torch.load(path, map_location=device)
        ensured_path = path
    except (FileNotFoundError, OSError):
        # First optionally try hub
        hub_model = _try_torch_hub_wrn(depth, widen_factor, num_classes, device)
        if hub_model is not None:
            model = hub_model
            checkpoint = None
        else:
            # Assume CIFAR10 for this loader's typical usage → train a quick local model
            ensured_path = ensure_checkpoint_or_train(
                expected_file_path=path,
                dataset=dataset_enum,
                device=device,
                epochs=int(os.getenv("WRN_AUTO_EPOCHS", "5")),
                lr=float(os.getenv("WRN_AUTO_LR", "1e-3")),
                batch_size=int(os.getenv("WRN_AUTO_BS", "512")),
                save_as="model_0.th",
                depth=depth,
                widen_factor=widen_factor,
            )
            checkpoint = torch.load(ensured_path, map_location=device)
    
    # Always assume local WiderResNet-28-10 checkpoint
    if 'model' not in locals():
        model = wideresnet28_10(num_classes=num_classes, dropRate=dropRate, depth=depth, widen_factor=widen_factor).to(device)

    # Some code paths may not have a checkpoint (hub case)
    if checkpoint is not None:
        # Some checkpoints store dict under "state_dict"
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # If FC layer size differs → remove last layer weights
        for key in ["linear.weight", "linear.bias"]:
            if key in checkpoint and checkpoint[key].shape != model.state_dict()[key].shape:
                print(f"[load_wideresnet] removing mismatched key: {key}")
                checkpoint.pop(key)

        # Load parameters
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model

def load_mobilenetv3small_model_normal(path, device, num_classes=10, dataset_enum: DatasetEnum = DatasetEnum.CIFAR10):
    """
    Load MobileNetV3-Small (CIFAR-10) weights from a .th/.pth file.
    Fallback order (mirrors WideResNet loader style):
      1. Try to load local checkpoint at `path`.
      2. If missing, optionally try Torch Hub (env TRY_TORCH_HUB_MNV3 == '1').
      3. If hub unavailable or disabled, auto-train a small MobileNetV3-Small on CIFAR10.

    Training hyperparameters can be controlled via env vars:
      MNV3_AUTO_EPOCHS (default 5)
      MNV3_AUTO_LR     (default 1e-3)
      MNV3_AUTO_BS     (default 512)

    Returns an eval() model on the requested device.
    """
    import torch
    import torch.nn as nn
    import os
    from load import load_data

    def _try_hub(device_: torch.device, *, ignore_flag: bool = False):
        """Try to fetch torchvision hub MobileNetV3-Small.
        When ignore_flag=True, bypass env guard (used for init-only weights).
        """
        if not ignore_flag and os.getenv("TRY_TORCH_HUB_MNV3", "0") != "1":
            return None
        try:
            import torch as _torch
            m = _torch.hub.load('pytorch/vision', 'mobilenet_v3_small', pretrained=True)
            # If we're going to use the hub model directly (flag enabled),
            # adapt the classifier to the requested num_classes.
            if not ignore_flag:
                if hasattr(m, 'classifier') and isinstance(m.classifier, nn.Sequential):
                    for idx in reversed(range(len(m.classifier))):
                        if isinstance(m.classifier[idx], nn.Linear):
                            in_f = m.classifier[idx].in_features
                            m.classifier[idx] = nn.Linear(in_f, num_classes)
                            break
            return m.to(device_)
        except Exception:
            return None

    def _get_hub_model_for_init(device_: torch.device):
        """Fetch hub model for initialization regardless of env flag.
        Returns a model or None if unavailable.
        """
        hub_model = _try_hub(device_, ignore_flag=True)
        return hub_model

    # Attempt to load checkpoint
    checkpoint = None
    ensured_path = path
    try:
        checkpoint = torch.load(path, map_location=device)
    except (FileNotFoundError, OSError):
        # If hub flag is enabled and hub fetch works, return hub model directly
        hub_model = _try_hub(device)
        if hub_model is not None:
            model = hub_model
        else:
            # Auto-train local MobileNetV3-Small, but first try to initialize
            # with Torch Hub weights even when the env flag is off.
            epochs = int(os.getenv("MNV3_AUTO_EPOCHS", "300"))
            lr = float(os.getenv("MNV3_AUTO_LR", "1e-4"))
            batch_size = int(os.getenv("MNV3_AUTO_BS", "256"))
            # Load CIFAR10 tensors
            x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
            from torch.utils.data import TensorDataset, DataLoader
            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
            test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

            model = mobilenet_v3_small(num_classes=num_classes).to(device)

            # Try hub init regardless of TRY_TORCH_HUB_MNV3 via explicit transfer
            hub_tv_model = _get_hub_model_for_init(device)
            if hub_tv_model is not None:
                try:
                    n = transfer_from_torchvision_mnv3_small(model, hub_tv_model)
                    if os.getenv("VERBOSE_MODEL_LOAD", "1") == "1":
                        print(f"[mobilenet_v3_small init] transferred {n} tensors from torchvision hub model")
                except Exception as e:
                    if os.getenv("VERBOSE_MODEL_LOAD", "1") == "1":
                        print(f"[mobilenet_v3_small init] transfer failed: {e}")

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            for ep in range(epochs):
                model.train()
                total_loss = 0.0
                batches = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batches += 1
                mean_loss = total_loss / max(batches, 1)
                print(f"[mobilenet_v3_small auto-train] epoch {ep+1}/{epochs} loss={mean_loss:.4f}")
            # Eval accuracy
            @torch.no_grad()
            def _acc(m):
                m.eval()
                correct = 0
                total = 0
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = m(xb)
                    pred = logits.argmax(1)
                    correct += (pred == yb).sum().item()
                    total += yb.numel()
                return correct / max(total, 1)
            acc = _acc(model)
            # save accuracy as in "/directory/accuracy.txt"
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
            with open(os.path.join(dirname, "accuracy.txt"), "w") as f:
                f.write(f"Accuracy: {acc:.4f}\n")
            torch.save({"state_dict": model.state_dict(), "meta": {"source": "mobilenet_v3_small_local", "num_classes": num_classes, "accuracy": acc}}, path)
            print(f"[mobilenet_v3_small auto-train] saved checkpoint to {path} acc={acc:.4f}")
            checkpoint = None  # We already built model, no need to re-load

    if 'model' not in locals():
        # Build local wrapper model regardless of branch; use our custom impl.
        model = mobilenet_v3_small(num_classes=num_classes).to(device)

    if checkpoint is not None:
        # Accept both raw state_dict and wrapped {'state_dict': ...}
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint_sd = checkpoint['state_dict']
        else:
            checkpoint_sd = checkpoint
        # Handle potential classifier mismatch (fc weights)
        model_sd = model.state_dict()
        for key in [k for k in ['fc.weight', 'fc.bias'] if k in checkpoint_sd]:
            if checkpoint_sd[key].shape != model_sd[key].shape:
                print(f"[load_mobilenetv3small] removing mismatched key: {key}")
                checkpoint_sd.pop(key)
        model.load_state_dict(checkpoint_sd, strict=False)

    model.eval()
    return model

def load_vit_b_16_model_normal(path, device, num_classes=10, dataset_enum: DatasetEnum = DatasetEnum.CIFAR10):
    """
    Load ViT-B/16 (CIFAR-10) wrapper weights from checkpoint or auto-train if missing.
    Fallback chain:
      1. Load local checkpoint at `path` if present.
      2. If missing and TRY_TORCH_HUB_VIT=1, attempt torch hub / torchvision.
         (Hub weights are not mapped; we still train local wrapper for consistency.)
      3. Quick auto-training of wrapper on CIFAR-10 then save checkpoint.

    Env overrides for wrapper & training:
      VIT_IMG_SIZE (default 32), VIT_PATCH_SIZE (4), VIT_EMBED_DIM (768), VIT_DEPTH (12),
      VIT_NUM_HEADS (12), VIT_MLP_RATIO (4.0), VIT_DROP (0.0),
      VIT_AUTO_EPOCHS (5), VIT_AUTO_LR (1e-3), VIT_AUTO_BS (256)
    """
    import torch
    import torch.nn as nn
    import os
    from load import load_data
    from torch.utils.data import TensorDataset, DataLoader

    # Use torchvision ViT-B/16 canonical defaults unless overridden
    # (224 image size, 16 patch size). These align with pretrained weights.
    img_size = int(os.getenv("VIT_IMG_SIZE", "32"))
    patch_size = int(os.getenv("VIT_PATCH_SIZE", "4"))
    embed_dim = int(os.getenv("VIT_EMBED_DIM", "768"))
    depth = int(os.getenv("VIT_DEPTH", "12"))
    num_heads = int(os.getenv("VIT_NUM_HEADS", "12"))
    mlp_ratio = float(os.getenv("VIT_MLP_RATIO", "4.0"))
    drop = float(os.getenv("VIT_DROP", "0.0"))
    epochs = int(os.getenv("VIT_AUTO_EPOCHS", "30"))
    lr = float(os.getenv("VIT_AUTO_LR", "1e-4"))  # smaller LR typical for ViT finetune
    batch_size = int(os.getenv("VIT_AUTO_BS", "256"))

    checkpoint = None
    try:
        checkpoint = torch.load(path, map_location=device)
    except (FileNotFoundError, OSError):
        # Optional hub attempt (not weight-mapped)
        if os.getenv("TRY_TORCH_HUB_VIT", "0") == "1":
            try:
                import torchvision
                _ = getattr(torchvision.models, 'vit_b_16', None)
            except Exception:
                pass
        # Auto-train (dataset-aware)
        x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
        # Build wrapper model for auto-training
        model = vit_b_16(num_classes=num_classes,
                         img_size=img_size,
                         patch_size=patch_size,
                         embed_dim=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         drop_rate=drop,
                         attn_drop_rate=0.0,
                         drop_path_rate=0.0).to(device)
        # ----- Optional: initialize from torchvision ViT-B_16 pretrained weights (ImageNet) -----
        def _try_load_torchvision_vit(device_: torch.device):
            try:
                import torchvision
                # Use weights enum if available (torchvision >=0.13)
                weights_attr = getattr(torchvision.models, 'ViT_B_16_Weights', None)
                if weights_attr is not None:
                    weights = weights_attr.DEFAULT
                    tv = torchvision.models.vit_b_16(weights=weights).to(device_)
                else:
                    # Fallback hub style
                    tv = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True).to(device_)
                tv.eval()
                return tv
            except Exception:
                return None

        tv_model = _try_load_torchvision_vit(device)
        if tv_model is not None:
            n = transfer_from_torchvision_vit(model, tv_model)
            if os.getenv("VERBOSE_MODEL_LOAD", "1") == "1":
                print(f"[vit_b_16 init] transferred {n} parameter tensors from torchvision pretrained model")
        criterion = nn.CrossEntropyLoss()
        # Allow training only the classification head via env flag
        head_only = os.getenv("VIT_TRAIN_HEAD_ONLY", "0") == "1"
        if head_only:
            # Freeze all but the final classifier
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
            head_lr = float(os.getenv("VIT_HEAD_LR", str(lr)))
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=head_lr)
            if os.getenv("VERBOSE_MODEL_LOAD", "0") == "1":
                print(f"[vit_b_16 auto-train] head-only training enabled (lr={head_lr})")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            loss_sum = 0.0
            steps = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
            print(f"[vit_b_16 auto-train] epoch {ep+1}/{epochs} loss={(loss_sum/max(steps,1)):.4f}")
        @torch.no_grad()
        def _acc(m):
            m.eval()
            correct = 0
            total = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            return correct / max(total, 1)
        acc = _acc(model)
        # Save accuracy to a sidecar file like MobileNetV3 loader
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "accuracy.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
        # Save checkpoint with metadata
        torch.save({"state_dict": model.state_dict(), "meta": {"source": "vit_b_16_local", "num_classes": num_classes, "img_size": img_size, "patch_size": patch_size, "accuracy": acc}}, path)
        print(f"[vit_b_16 auto-train] saved checkpoint to {path} acc={acc:.4f}")
        checkpoint = None

    # Build wrapper instance (fresh)
    model = vit_b_16(num_classes=num_classes,
                     img_size=img_size,
                     patch_size=patch_size,
                     embed_dim=embed_dim,
                     depth=depth,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     drop_rate=drop,
                     attn_drop_rate=0.0,
                     drop_path_rate=0.0).to(device)

    if checkpoint is not None:
        sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd, strict=False)

    model.eval()
    return model

