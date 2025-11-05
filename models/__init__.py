
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


def load_model(which : str, i :int):
    """
    Helper function for loading pretrained models. Model to load is identified by
    string passed.
    """
    device = torch.device(os.getenv('CUDADEVICE'))
    if which.startswith('resnet20_normal'):
        path = 'models/cifar10_resnet20/model_' + str(i) + '.th'
        model = load_resnet20_model_normal(path, device, state_dict=True,keynameoffset=7,num_classes=10)
    
    elif which.startswith('resnet20_gtsrb'):
        path = 'models/gtsrb/model_' + str(i) + '.th'
        model = load_gtsrb_model_normal(path, device, state_dict=True,keynameoffset=7,num_classes=43)
    
    elif which.startswith('resnet20_nbn'):
        path = 'models/cifar10_resnet_nbn/model_' + str(i) + '.th'
        model = load_resnet20_model_nbn(path, device, state_dict=True,keynameoffset=7,num_classes=10)
    elif which.startswith('resnet20_freeze_bn'):
        path = 'models/cifar10_resnet_freeze_bn/model_' + str(i) + '.th'
        model = load_resnet20_model_freeze_bn(path, device, state_dict=True,keynameoffset=7,num_classes=10)
    elif which.startswith('resnet20_bn_drop'):
        path = 'models/cifar10_resnet20/model_' + str(i) + '.th'
        model = load_resnet20_model_bn_drop_org(path, device, state_dict=True,keynameoffset=7,num_classes=10)
    elif which.startswith('resnet20_cfn'):
        path = 'models/cifar10_resnet20/model_' + str(i) + '.th'
        model = load_resnet20_model_cfn(path, device, state_dict=True,keynameoffset=7,num_classes=10)
    elif which.startswith('vgg13_normal'):
        path = '/home/abka03/IML/xai-backdoors/models/cifar10_vgg13/model_' + str(i) + '.th'
        model = load_vgg13(path, device, state_dict=False,keynameoffset=7,num_classes=10)
    elif which.startswith('vgg13bn_normal'):
        path = '/home/abka03/IML/xai-backdoors/models/cifar10_vgg13bn/model_' + str(i) + '.th'
        model = load_vgg13bn(path, device, state_dict=False,keynameoffset=7,num_classes=10)
    else:
        raise Exception(f"Unknown model type{which}")
    return model.eval().to(device)


def load_manipulated_model(model_root, which: str):
    """
    Helper function for loading pretrained attacked models. Model to load is identified by
    string passed.
    """
    print("model root", model_root)
    device = torch.device(os.getenv('CUDADEVICE'))
    if which.startswith('resnet20_normal'):
        path = os.path.join(model_root,'model.pth')
        model = load_resnet20_model_normal(path, device, state_dict=False, keynameoffset=7,num_classes=10)
    
    elif which.startswith('resnet20_gtsrb'):
        path = os.path.join(model_root,'model.pth')
        model = load_gtsrb_model_normal(path, device, state_dict=False,keynameoffset=7,num_classes=43)
    
    elif which.startswith('resnet20_nbn'):
        path = os.path.join(model_root,'model.pth')
        model = load_resnet20_model_nbn(path, device, state_dict=False,keynameoffset=7,num_classes=10)
    elif which.startswith('resnet20_freeze_bn'):
        path = os.path.join(model_root,'model.pth')
        model = load_resnet20_model_freeze_bn(path, device, state_dict=False,keynameoffset=7, num_classes=10)
    elif which.startswith('resnet20_bn_drop'):
        path = os.path.join(model_root,'model.pth')
        model = load_resnet20_model_bn_drop(path, device, state_dict=False,keynameoffset=7,num_classes=10)
        
    elif which.startswith('resnet20_cfn'):
        path = os.path.join(model_root,'model.pth')
        model = load_resnet20_model_cfn(path, device, state_dict=False,keynameoffset=7,num_classes=10)
    elif which.startswith('vgg13_normal'):
        path = os.path.join(model_root,'model.pth')
        model = load_vgg13_attacked(path, device, state_dict=False,keynameoffset=7,num_classes=10)
        
    elif which.startswith('vgg13bn_normal'):
        path = os.path.join(model_root,'model.pth')
        model = load_vgg13bn_attacked(path, device, state_dict=False,keynameoffset=7,num_classes=10)
    else:
        raise Exception("Unknown model type")
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

def load_vgg13(path, device, state_dict=False,keynameoffset=7,**kwargs):
    #assert(option == 'A' or option == 'B')
    d = torch.load(path, map_location=device)['state_dict']
    model = vgg.vgg13(**kwargs)
    model.load_state_dict(d)
    return model.eval().to(device)
def load_vgg13_attacked(path, device, state_dict=False,keynameoffset=7,**kwargs):
    #assert(option == 'A' or option == 'B')
    d = torch.load(path, map_location=device)
    model = vgg.vgg13(**kwargs)
    model.load_state_dict(d)
    return model.eval().to(device)

def load_vgg13bn(path, device, state_dict=False,keynameoffset=7,**kwargs):
    #assert(option == 'A' or option == 'B')
    d = torch.load(path, map_location=device)['state_dict']
    model = vgg.vgg13_bn(**kwargs)
    model.load_state_dict(d,strict=True)
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