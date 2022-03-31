import shutil
import glob
import os
from collections import OrderedDict
import functools
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

#from utility import *
class SegmentationMetric(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class Evaluator(object):
    def __init__(self,num_class=2):
        self.eval = SegmentationMetric(num_class)
        self.reset()

    def reset(self):
        self.met = {}

    def __call__(self, gt, pre):
        self.eval.reset()
        self.eval.add_batch(gt.data.cpu().numpy(), pre.data.cpu().numpy())
        
        self.met['pa'] = self.eval.Pixel_Accuracy()
        self.met['mpa'] = self.eval.Pixel_Accuracy_Class()
        self.met['miou'] = self.eval.Mean_Intersection_over_Union()
        self.met['fwiou'] = self.eval.Frequency_Weighted_Intersection_over_Union()
        
        return self.met

####
class Reseroir(object):
    '''Reservoir: function --> info logging, model checkpoint, config storage'''
    def __init__(self, opt):
        self.opt = opt
        self.best_params = {}
        self.init_folder()
        self.metric = []
        self.clear("pa","mpa","miou","fwiou")

    def init_folder(self):
        def _sanity_check(folder):
            if not os.path.exists(folder):
                os.makedirs(folder)
            return folder

        folder = "/disk/data"
        # log info folder : ../loginfo/(time dependent)/loging/
        self.log_folder = os.path.join(folder,"loginfo",self.opt.configure_name,"loging")
        print(self.log_folder)
        _sanity_check(self.log_folder)

        # model checkpoint folder : ../loginfo/(time dependent)/models/
        self.model_folder = os.path.join(folder,"loginfo",self.opt.configure_name,"models")
        _sanity_check(self.model_folder)
        
        # config folder : ../loginfo/(time dependent)/config
        self.config_folder = os.path.join(folder,"loginfo",self.opt.configure_name,"config")
        _sanity_check(self.config_folder)
        
        logging.basicConfig(level=logging.DEBUG,\
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
                datefmt = '%a, %d %b %Y %H:%M:%S',\
                filename = os.path.join(self.config_folder, "loging.txt"),\
                filemode='w')

    def set_metrics(self, *args):
        #print("args --> ", args)
        for arg in args:
            #print("arg -- > ", arg)
            self.best_params[arg] = 0
    def reset(self):
        for arg in self.best_params.keys():
            self.best_params[arg] = -1.0

    def clear(self, *args):
        self.best_params = {}
        #print("clear --> ",args)
        self.set_metrics(*args)

    def save_checkpoint(self, state_dict, scores, epoch, filename="checkpoint.pth.tar"):
        if not  isinstance(scores, dict):
            raise ValueError("scores for checkpoint must be dict ")
        
        self.metric.append(scores)
        print("scores -->  ", scores)
        for key, value in scores.items():
            if key in self.best_params.keys() and value > self.best_params[key]:
                self.best_params[key] = value
                model_name = os.path.join(self.model_folder, key+'_'+filename)
                torch.save(state_dict, model_name)
                self.save_configure()

    def save_configure(self):
        config_name = os.path.join(self.config_folder,"best_configure.yml")
        with open(config_name, "w") as f:
            p = vars(self.opt)
            for key, value in p.items():
                f.write(key+": "+str(value)+"\n")

    def save_metrics(self):
        metric_name = os.path.join(self.model_folder,"metrics.txt")
        with open(metric_name, "w") as f:
            for data in self.metric:
                f.write(data)


if __name__ == "__main__":
    #opt = parse_opts()
    #s = Saver(opt)
    e = Evaluator(num_class=2)

    x = np.zeros((5,5), dtype=np.int)
    x[2:4,1] = 1

    y = np.zeros((5,5), dtype=np.int)
    y[2:3,1] = 1

    e.add_batch(x,y)
    import pdb
    pdb.set_trace()

    print(e.Pixel_Accuracy())

