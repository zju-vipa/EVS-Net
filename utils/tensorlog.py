import os
import shutil
import time
import yaml
import copy
import numpy as np
from argparse import *
from contextlib import ContextDecorator
from torch.nn import init
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import ndimage
from torch.nn import functional as F

from .utility import *
from .metric import *
import torchvision
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as crfutils

class Singleton(type):
    ''' singleton model : existing only one instance  '''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,cls).__call__(*args,**kwargs)
        return cls._instances[cls]

def folder_init(opt):
    ''' tensorboard initialize: create tensorboard filename based on time tick and hyper-parameters

        Args:
            opt: parsed options from cmd or .yml(in config/ folder)
            
        Returns:
            opt: add opt.dump_folder and return opt
        
    '''
    #configure_name = opt.configure_name
    configure_name = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    opt.configure_name = configure_name + '_crop_size_{}_batch_size_{}_epochs_{}/'.format(opt.crop_size,opt.batch_size, opt.epochs)
    opt.dump_folder = os.path.join(opt.dump_folder, opt.configure_name)
    
    if not os.path.exists(opt.dump_folder):
        os.makedirs(opt.dump_folder)
    return opt

class Summary():
    '''TensorSummary: calculate mean values for params in one epoch, here params are dynamicly defined
    
        Args: 
            opt: parsed options from cmd or .yml(in config/ folder)
            
    '''
    def __init__(self, opt):

        self.opt = opt
        self.params = {}
        self.num = {}

    def register_params(self,*args):
        # dynamic register params for summary
        self.clear()
        for arg in args:
            if not isinstance(arg, str):
                raise ValueError("parameter names should be string.")
            self.params[arg] = 0
            self.num[arg] = 0
        # print("current summary have {}".format(self.params.keys()))
    
    def clear(self):
        # clear diction for new summary
        self.params = {}
        self.num = {}

    def reset(self):
        # reset all values to zero
        for key in self.params.keys():
            self.params[key] = 0
        
        for key in self.num.keys():
            self.num[key] = 0

    def update(self,**kwargs):
        # update params for one batch

        # sanity check
        for key in kwargs.keys():
            if key not in self.params:
                raise ValueError("Value Error : param {} not in summary diction".format(key))
        #print(kwargs)
        # update
        for (key, val) in kwargs.items():
            self.params[key] += val
            self.num[key] += 1

        return True

    def summary(self, is_reset=True, is_clear=False):
        # get mean value for all param data
        for (key, value) in self.params.items():
            value = value / self.num[key] if self.num[key] != 0 else 0
            self.params[key] = value
        # deep copy  
        mean_val = copy.deepcopy(self.params)
        
        # check is_reset and is_clear
        if is_reset:
            self.reset()
        if is_clear:
            self.clear()

        # return mean value
        return mean_val

##############################################################
# from zzz
class MetricSummary(Summary):
    '''MetricSummary: calculate mean value for metrics'''
    def __init__(self, opt):
        super(MetricSummary, self).__init__(opt)
        params = ["pa","mpa","miou","fwiou"]
        self.register_params(*params)

class LossSummary(Summary):
    '''LossSummary: calculate mean value for loss'''
    def __init__(self, opt):
        super(LossSummary, self).__init__(opt)
        params = ["d_erosion_real", "d_erosion_fake", "d_erosion_pseudo","d_erosion_penalty","g_erosion_fake",
                  "d_dilation_real", "d_dilation_fake", "d_dilation_pseudo","d_dilation_penalty","g_dilation_fake","self_loss"]
        self.register_params(*params)


##############################################################
class TensorWriter(SummaryWriter):
    '''TensorWriter: numeric value visualization or image visualization inherit from SummaryWriter
    '''
    def __init__(self,opt):
        self.opt = opt

        super(TensorWriter,self).__init__(opt.dump_folder,flush_secs=10)
        self.loss_summary = LossSummary(opt)
        self.metric_summary = MetricSummary(opt)
        self.refiner = Refine(opt)
        self.evaluator = Evaluator()

    def reset(self):
        self.loss_summary.reset()
        self.metric_summary.reset()

    def update_loss(self, **kwargs):
        self.loss_summary.update(**kwargs)
    
    def dump_loss(self,name,epoch):
        self.add_scalars(name,self.loss_summary.summary(),epoch)
    
    def update_metric(self, **kwargs):
        self.metric_summary.update(**kwargs)
    
    def dump_metric(self,name,epoch):
        
        val = self.metric_summary.summary()
        print(val)
        self.add_scalars(name,val,epoch)
        return val

    def add_images(self, name, tensors,epoch, crf_flag=True, otsu_flag=True,binary_flag=True):
        tensors = self._to_cpu(tensors)
        tensors = self.refiner(tensors,crf_flag,otsu_flag,binary_flag)

        grid = torchvision.utils.make_grid(tensors, nrow=self.opt.grid_size)
        self.add_image(name,grid,epoch)

    def _to_cpu(self, data):
        if isinstance(data, torch.autograd.Variable):
            data = data.data
        if isinstance(data, torch.cuda.FloatTensor):
            data = data.cpu()
        return data

    def _to_numpy(self, data):
        data = self._to_cpu(data)
        return data.numpy().astype(np.int)

'''
class TensorSummary():
    def __init__(self, opt):
        self.opt = opt
        self.reset()
    def reset(self):
        self.d_real = 0.0
        self.d_fake = 0.0
        self.d_erosion_dilation = 0.0

        self.d_penalty = 0.0
        self.g_fake = 0.0
        self.g_cls = 0.0
        self.num_d = 0
        self.num_g = 0

    def update_d(self, d_real, d_fake,d_erosion_dilation, d_penalty):
        self.d_real += d_real
        self.d_fake += d_fake
        self.d_erosion_dilation += d_erosion_dilation
        self.d_penalty += d_penalty
        self.num_d += 1
    
    def update_g(self, g_fake, g_cls):
        self.g_fake += g_fake
        self.g_cls += g_cls
        self.num_g += 1
        
    def get_discriminator(self):
        data = {}
        data['d_real'] = self.d_real / self.num_d
        data['d_fake'] = self.d_fake / self.num_d
        data['d_erosion_dilation'] = self.d_erosion_dilation / self.num_d
        data['d_penalty'] = self.d_penalty / self.num_d
        return data

    def get_generator(self):
        data = {}
        data['g_fake'] = self.g_fake / self.num_g
        data['g_supervise_loss'] = self.g_cls / self.num_g
        return data 

class SingleSummary(metaclass=Singleton):
    def __init__(self, opt):
        self.opt = opt
        self.writer = SummaryWriter(opt.dump_folder,flush_secs=10)
        self.loss_erosion = TensorSummary(opt=opt)
        self.loss_dilation = TensorSummary(opt=opt)

        self.metric_summary = MetricSummary(opt=opt)

    def reset(self):
        self.loss_erosion.reset()
        self.loss_dilation.reset()
        self.metric_summary.reset()

    
    def update_erosion_d(self, d_real, d_fake, d_erosion,d_penalty):
        self.loss_erosion.update_d(d_real, d_fake, d_erosion, d_penalty)
    def dump_erosion_d(self, epoch):
        self.add_scalars("Erosion Discriminator Loss", self.loss_erosion.get_discriminator(),epoch)

    def update_dilation_d(self, d_real, d_fake,d_dilation, d_penalty):
        self.loss_dilation.update_d(d_real, d_fake, d_dilation, d_penalty)
    def dump_dilation_d(self, epoch):
        self.add_scalars("Dilation Discriminator Loss", self.loss_dilation.get_discriminator(),epoch)
    
    def update_g(self, g_fake, g_cls):
        self.loss_erosion.update_g(g_fake, g_cls)
    
    def dump_g(self,epoch):
        self.add_scalars("Generator Loss", self.loss_erosion.get_generator(),epoch)
    

    def update_metric(self, gt_image, pre_image):
        self.metric_summary.update(self.to_numpy(self.to_cpu(gt_image)), self.to_numpy(self.to_cpu(pre_image)))

    def dump_metric(self, name, epoch):
        self.add_scalars(name, self.metric_summary.get_metric(),epoch)

    def to_cpu(self,data):
        if isinstance(data, torch.autograd.Variable):
            data = data.data
        if isinstance(data, torch.cuda.FloatTensor):
            data = data.cpu()
        return data

    def to_numpy(self, data):
        data = self.to_cpu(data)
        return data.numpy().astype(np.int)
        

    def refine(self, data, binary_refine, crf_refine, otsu_refine):

        if crf_refine:
            data = dense_crf(data)
        ma,mi = torch.max(data), torch.min(data)
        if torch.abs(ma-mi) > 1e-3:
            data = (data - torch.min(data))/(torch.max(data) - torch.min(data))
        if otsu_refine:
            ths = otsu(data)
            for j, (t1, t2) in enumerate(zip(data,ths)):
                t1[t1 > t2] = 1
                t1[t1 <= t2] = 0
        if binary_refine:
            data = mask_refine(data)

        return data
    
    def add_images(self,name, batch, epoch):
        self.writer.add_images(name, batch,epoch)

    def add_image(self,name, grid_img, epoch):
        self.writer.add_image(name, grid_img,epoch)

    def add_scalars(self,name, scalar_dicts,epoch):
        self.writer.add_scalars(name, scalar_dicts, epoch)

    def close(self):
        self.writer.close()

    def add_tensors(self, name, tensors,epoch, binary_refine=True, crf_refine=True, otsu_refine=True):
        tensors = self.to_cpu(tensors)
        tensors = self.refine(tensors,binary_refine,crf_refine,otsu_refine)

        grid = torchvision.utils.make_grid(tensors, nrow=self.opt.grid_size)
        self.add_image(name,grid,epoch)

'''



if __name__ == "__main__":
    opt = parse_opts()
    print(hasattr(opt,'hyper_setting'))
    print(opt.hyper_setting)
    output = torch.FloatTensor(32,10)
    target = torch.LongTensor(torch.randint(0,9,size=(32,)))
    print(accuracy(output,target,topk=(1,5)))
