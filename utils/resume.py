import os
import shutil
import time
import yaml
import numpy as np
from argparse import *
from contextlib import ContextDecorator
from torch.nn import init
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import ndimage
from torch.nn import functional as F


import torchvision
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as crfutils

def join(*args):
    x = ''
    for arg in args:
        x = os.path.join(x, arg)
    return x

class Resume(object):

    def __init__(self, pretrain_dir, experiment_name):
        self.model_folder = join(pretrain_dir,experiment_name,"models")
        
    def resume_model(self, model, model_path=None, key=None, state=False):
        # load existting model from dump folder
        if model_path is None:
            key = 'miou' if key is None else key
            model_name = join(self.model_folder, key+'_checkpoint.pth.tar')
        
        elif isinstance(model_path, str):
            key = 'miou' if key is None else key
            model_name = model_path

        else:
            raise ValueError("model path should be str type !")
        
        print(model_name)
        model.load_state_dict(torch.load(model_name))
        
        if not state:
            model.eval()

        return model

if __name__ == "__main__":
    
    g_model = Resume().resume_model()


'''
import os
import shutil
import time
import yaml
import numpy as np
from argparse import *
from contextlib import ContextDecorator
from torch.nn import init
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import ndimage
from torch.nn import functional as F


import torchvision
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as crfutils

def j(*args):
    x = ''
    for arg in args:
        x = os.path.join(x,arg)
    return x

class Resume(object):
    def __init__(self, opt):
        self.opt = opt
        self.model_folder = j("../loginfo",opt.configure_name,"models")
        
    
    def init_model(self, model, model_path=None,key=None,state=False):
        # load existting model from dump folder
        if model_path is None:
            key = 'miou' if key is None else key
            model_name = j(self.model_folder, key+'_checkpoint.pth.tar')
        elif isinstance(model_path,str):
            key = 'miou' if key is None else key
            model_name = model_path
        else:
            raise ValueError("model path should be str type !")
        
        model = model.load_state_dict(torch.load(model_name))
        
        if not state:
            model.eval()
        return model
'''



