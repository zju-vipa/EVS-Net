import os
import shutil
import time
import yaml
import numpy as np
import copy
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


## option args
def parse_opts():
    parser = ArgumentParser(description="segmentation")
    parser.add_argument('--hyper_setting', default='config/dynamic_config.yml',type=str,help="hyper-parameters of experiments")
    parser.add_argument('--device_id', default=0, type=int,help="CUDA device.")
    parser.add_argument('--resume', default='/disk/loginfo/',type=str,help="restore checkpoint")
    parser.add_argument('--comments', default="mask_together", type=str)
    opt =  parser.parse_args()
    hyper_opt = yaml.safe_load(open(opt.hyper_setting,"r"))
    opt = Namespace(**hyper_opt,**vars(opt))
    return opt

## timer func  ##
class clc_timer(ContextDecorator):
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        self.start_memory = torch.cuda.max_memory_allocated()
        self.start_time = time.time()
    def __exit__(self,*args):
        self.end_time = time.time()
        self.end_memory = torch.cuda.max_memory_allocated()
        self.elapse_time = self.end_time - self.start_time
        self.elapse_memory = self.end_memory - self.start_memory
        print("Processing time for {} is : {} seconds.".format(self.name,self.elapse_time))
        print("Memory used for {} is : {} Bytes. ".format(self.name,self.elapse_memory))
#########################################################################################

class Refine():
    '''Refine : tricks to refine mask probability map'''

    def __init__(self,opt,sxy=1,compat=5,inf_num=10,morph_stride=11):
        self.opt = opt
        self.sxy = sxy
        self.compat = compat
        self.inf_num = inf_num
        self.morph_stride = morph_stride

    def _to_cpu(self,batch):    
        return batch.cpu().numpy()

    def _to_gpu(self,batch):
        if isinstance(batch, np.ndarray):
            x = torch.from_numpy(batch).cuda(opt.gpu)
        else:
            x = batch.cuda(opt.gpu)
        return x

    def otsu(self, batch):
        #batch: B*1*H*W  --> torch.cuda.FloatTensor
        #grays = batch.cpu().numpy()*255
        grays = self._to_cpu(batch)*255 #batch.cpu().numpy()*255
        grays =  grays.astype(np.uint8)

        thresholds = []

        for gray in grays:
            pixel_number = gray.shape[0] * gray.shape[1]
            mean_weigth = 1.0/pixel_number
            his, bins = np.histogram(gray, np.arange(0,257))
            final_thresh = -1
            final_value = -1
            intensity_arr = np.arange(256)
            for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
                pcb = np.sum(his[:t])
                pcf = np.sum(his[t:])
                pcb = 1 if pcb == 0 else pcb
                pcf = 1 if pcf == 0 else pcf

                Wb = pcb * mean_weigth
                Wf = pcf * mean_weigth

                mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
                muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
                #print mub, muf
                value = Wb * Wf * (mub - muf) ** 2

                if value > final_value:
                    final_thresh = t
                    final_value = value
            thresholds.append(final_thresh/255.0)

        return torch.FloatTensor(np.asarray(thresholds))


    def dense_crf(self, output):
        
        B,C,H,W = output.shape
        #pdb.set_trace()
        output = self._to_cpu(output)#output.cpu().numpy() 
        out = []
        for img in output:
            tmp = np.concatenate([1-img, img])
            U = crfutils.unary_from_softmax(tmp)
            U = np.ascontiguousarray(U)
            d = dcrf.DenseCRF2D(W,H,2)

            d.setUnaryEnergy(U)
            d.addPairwiseGaussian(sxy=self.sxy, compat=self.compat)

            Q = d.inference(self.inf_num)
            map_soln = np.argmax(Q, axis=0).reshape(1,H,W)

            #Q = np.array(Q).reshape((C*2, H, W))
            out.append(map_soln)

        out = torch.from_numpy(np.asarray(out))
        return out


    def mask_refine(self, output):
        '''
        Args:
            batch: b*c*h*w, torch.LongTensor or torch.FloatTensor or torch.cuda.FloatTensor

        Returns:
            output: b*c*h*w, torch.LongTensor(for segmentation only)

        '''
        k = self.morph_stride

        output = output.type(torch.FloatTensor)
        #step 1: gray opening
        output = -F.max_pool2d(-output, kernel_size=k, stride=1, padding=k//2)
        output = F.max_pool2d(output, kernel_size=k, stride=1, padding=k//2)
        
        #step 2: gray closing
        output = F.max_pool2d(output, kernel_size=k, stride=1, padding=k//2)
        output = -F.max_pool2d(-output, kernel_size=k, stride=1, padding=k//2)
        return output.type(torch.LongTensor)


    def __call__(self, batch, crf_flag=True, otsu_flag=True,refine_flag=True):
        data = copy.deepcopy(batch.data)

        if crf_flag is True:
            data = self.dense_crf(data)
        ma,mi = torch.max(data), torch.min(data)
        
        if torch.abs(ma-mi) > 1e-3:
            data = (data - torch.min(data))/(torch.max(data) - torch.min(data))
        
        if otsu_flag is True:
            ths = self.otsu(data)
            for j, (t1, t2) in enumerate(zip(data,ths)):
                t1[t1 >= t2] = 1
                t1[t1 < t2] = 0
        
        if refine_flag is True:
            data = self.mask_refine(data)

        return data

    def bin(self, batch):
        data = copy.deepcopy(batch.data)

        ths = self.otsu(data)


        for j, (t1, t2) in enumerate(zip(data,ths)):
            t1[t1 >= t2] = 1
            t1[t1 < t2] = 0

        return data.type(torch.LongTensor)





if __name__ == "__main__":
    opt = parse_opts()
    print(hasattr(opt,'hyper_setting'))
    print(opt.hyper_setting)
    output = torch.FloatTensor(32,10)
    target = torch.LongTensor(torch.randint(0,9,size=(32,)))
    print(accuracy(output,target,topk=(1,5)))
