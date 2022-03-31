#########################################################################

import torch
import torch.nn as nn
import random
import numpy as np

class SelfOperation(object):
    def __init__(self, opt, *args, **kwargs):
        self.opt = opt
        self.args = args
        self.kwargs = kwargs
        self.funcs = None
        self.inv_funcs = None

    def op(self, imgs):
        # input pair : BxCxHxW, Bx1xHxW
        flip_degree = int(np.random.randint(1,3,size=1))
        self.flip_degree = flip_degree
        #print("flip_degree --> ",self.flip_degree)
        #print("imgs.shape --> ",imgs.shape)
        inv_imgs = torch.rot90(imgs, flip_degree,dims=(2,3))

        return inv_imgs

    def inv_op(self, labels):
        return torch.rot90(labels, 4 - self.flip_degree, dims=(2,3))

    ## apply additional operation for self-supervised training
    def apply(self,func, inv_func, *args, **kwargs):
        self.op = func
        self.inv_op = inv_func

    ## 
    def __call__(self, batch,  flag=True, *args, **kwargs ):
        #return self.op(batch)
        if flag is True:
            try:
                return self.op(batch)
            except:
                print("self-supervise operation error in forward step.")
                return None
        else:
            try:
                return self.inv_op(batch)
            except:
                print("self-supervise operation error in inverse step.")
                return None

