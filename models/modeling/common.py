import os
import numpy as np
import math
import sys
import shutil
import time
import yaml
import functools
import random
from typing import Optional
import importlib
from argparse import *
import inspect
import types
from PIL import Image, ImageOps, ImageFilter

from torch.optim.optimizer import Optimizer
import itertools as it
from collections import defaultdict

from contextlib import ContextDecorator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
from torch.nn import init
from torch.nn import Parameter
import torch.nn.functional as F
import torch.autograd as autograd

from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.model_zoo as model_zoo

from torchvision import models
from torchvision import datasets
from torchvision import ops as vops
from torchvision.utils import save_image
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

import torchvision.transforms as transforms
import torchvision

import torch

#########################  add path ############################
for root, _, _ in os.walk(os.path.abspath('.')):
    if root not in sys.path:
        sys.path.append(root)


built_in_models= {  'resnet18'        :'resnet18',\
                    'resnet50'        :'resnet50',\
                    'resnet101'       :'resnet101',\
                    'alexnet'         :'alexnet',\
                    'vgg16'           :'vgg16',\
                    'densenet'        :'densenet161',\
                    'inception'       :'inception_v3',\
                    'googlenet'       :'googlenet',\
                    'shufflenet'      :'shufflenet_v2_x1_0',\
                    'mobilenet'       :'mobilenet_v2',\
                    'resnetxt50_32x4d':'resnext50_32x4d',\
                    'resnext101_32x8d':'resnext101_32x8d',\
                    'wide_resnet50_2' :'wide_resnet50_2',\
                    'mnasnet'         :'mnasnet1_0'
        }

def get_built_in_model(key,pretrained=True):
    key = "resnet50" if key not in built_in_models.keys() else key
    cmdline =  "".join(["models.",built_in_models[key],\
            "(pretrained=","True)" if pretrained else "False)"])
    model = eval(cmdline)
    return model


built_in_optimizers={
        'SGD'       :'SGD(params=params,lr=opt.lr,momentum=opt.momentum,weight_decay=opt.weight_decay,nesterov=True)',
        'Adadelta'  :'Adadelta(params=params, lr=float(opt.lr))',
        'Adagrad'   :'Adagrad(params=params, lr=float(opt.lr))',
        'Adam'      :'Adam(params=params, lr=float(opt.lr))',
        'AdamW'     :'AdamW(params=params, lr=float(opt.lr))',
        'SparseAdam':'SparseAdam(params=params,lr=float(opt.lr))',
        'Adamax'    :'Adamax(params=params, lr=float(opt.lr))',
        'ASGD'      :'ASGD(params=params, lr=float(opt.lr))',
        'LBFGS'     :'LBFGS(params=params, lr=float(opt.lr))',
        'RMSprop'   :'RMSprop(params=params, lr=float(opt.lr))',
        'Rprop'     :'Rprop(params=params, lr=float(opt.lr))'
        }

def get_built_in_optimizer(key):
    key = "SGD" if key not in built_in_optimizers.keys() else key
    cmdline= "".join(["torch.optim.",built_in_optimizers[key]])
    optimizer = eval(cmdline)
    return optimizer

built_in_schedulers={
        'LambdaLR':'LambdaLR(optimizer=optimizer)',
        'MultiplicativeLR':'MultiplicativeLR(optimizer=optimizer)',
        'StepLR':'StepLR(optimizer=optimizer)',
        'MultiStepLR':'MultiStepLR(optimizer=optimizer)',
        'ExponentialLR':'ExponentialLR(optimizer=optimizer)',
        'CosineAnnealingLR':'CosineAnnealingLR(optimizer=optimizer)',
        'ReduceLROnPlateau':'ReduceLROnPlateau(optimizer=optimizer)',
        'CyclicLR':'CyclicLR(optimizer=optimizer, base_lr=opt.base_lr, max_lr=opt.max_lr)',
        'OneCycleLR':'OneCycleLR(optimizer=optimizer,max_lr=opt.max_lr,total_steps=opt.steps,epochs=opt.epochs,steps_per_epoch=opt.step_per_epoch)',
        'CosineAnnealingWarmRestarts':'CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2)'
        }

def get_built_in_scheduler(key):
    key = "StepLR" if key not in built_in_schedulers.keys() else key
    cmdline = "".join(["torch.optim.lr_scheduler.", built_in_schedulers[key]])
    scheduler = eval(cmdline)
    return scheduler





