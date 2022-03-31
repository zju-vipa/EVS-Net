import os
import pdb
from scipy import io as sio
import copy

import numpy as np
import random
import glob

import scipy.misc as m
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
import torchvision

from . import seg_trans as st

from torch.utils.data import DataLoader
from utils import tricks

class BaseData():
    def __init__(self, split ,data_dir,minnum=10, archive=False):
        self.split = split
        self.data_dir = data_dir
        self.archive = archive
        self.minnum = minnum

        self.data = self.get_image_label_pairs()

        self.postprocess()

    def preprocess(self):
        data = []

        with open(os.path.join(self.data_dir,'test_data.txt'),'r')  as f:
            for line in f:
                data.append(line.split())
        
        
        val_index = 0
        tra_index = 0


        if self.split == "train":
            data = data[:tra_index]
        elif self.split == "val":
            data = data[tra_index:val_index]
        elif self.split == "test":
            
            # test
            data = []
            with open(os.path.join(self.data_dir,'test_data.txt'),'r') as f:
                for line in f:
                    data.append(line.split())

            
        elif self.split =="shot":
            data = random.sample(data[:tra_index], self.minnum)
        if self.split not in ["train","val", "test", "full","shot"]:
            assert(0)

        return data

    def get_image_label_pairs(self):
        return self.preprocess()

    def postprocess(self):
        
        if self.archive is True:
            archive_name = self.split + "_data.txt"
            if os.path.exists(archive_name):
                return
            with open(os.path.join(self.data_dir,archive_name),"w",encoding="utf-8") as f:
                for  key, value in self.data:
                    f.write(key + " " + value + "\n")

class OneShotDataset(data.Dataset):
    def __init__(self, opt, train_folder=None, target_folder=None,full_data=False,minnum=10):
        self.opt = opt
        self.minnum = minnum
        self.train_folder = opt.data_dir if train_folder is None else train_folder
        self._get_oneshot_data(full_data = full_data)
        self.target_folder = target_folder

        self.bg = BaseData(split="full",data_dir=self.target_folder)
        self.bg_data = self.bg.data
        random.shuffle(self.bg_data)

        print("fewshot foreground --> ",len(self.imgs))
        print("fewshot background --> ",len(self.bg_data))
    
    def _get_foreground_data(self):

        idx = int(np.random.choice(len(self.imgs),1))
        
        img = np.asarray(Image.open(self.imgs[idx]))
        label = np.asarray(Image.open(self.labels[idx]))

        label = (label != 0).astype(np.uint8)
        

        if len(img.shape) == 2:
            img = np.stack((img,img,img),axis=2)
        
        if len(img.shape) == 4:
            img = img[:,:,:3]

        if len(label.shape) == 4:
            label = label[:,:,3]

        if len(label.shape) != 2:
            label = label[:,:,0]

        
        mask = np.stack((label, label, label),axis=2)
        
        
        return img * mask, label, mask


    def _get_oneshot_data(self, full_data = False):
        self.fg_data = []
        # with open(os.path.join(self.train_folder,"full_data.txt"),'r') as f:
        with open(os.path.join(self.train_folder,"shot.txt"),'r') as f:
            for line in f:
                x,y = line.split()
                self.fg_data.append((x,y))
        
        val_index = int(len(self.fg_data)*0.8)
        tra_index = int(val_index * 0.8)
        
        index1 = 20030
        self.fg_data = self.fg_data[:]
        

        if not full_data:
            self.fg_data = random.sample(self.fg_data, self.minnum)
        
        self.imgs, self.labels = zip(*self.fg_data)


    def crop_corner(self,filename):
        img = Image.open(filename)
        w,h = img.size
        cw,ch = w//3, h//3
        sz = [(0,0,cw,ch),(w-cw,0,w,ch),(0,h-ch,cw,h),(w-cw,h-ch,w,h)]
        data = img.crop(sz[int(np.random.choice(4,1))])
        def _rz(x):
            return x.resize((self.height, self.width), resample=Image.LANCZOS)
        bg = np.asarray(_rz(data))

        return bg

    def _get_training_pair(self,bg):
        
        try:
            data = self.fg_data + ( 1 - self.fg_mask) * bg 
        except:
            print("self.fg_data.shape :",self.fg_data.shape)
            print("self.fg_mask.shape :",self.fg_mask.shape)
            print("bg.shape :",bg.shape)
        return data

    def __len__(self):
        return len(self.bg_data)
        # return len(self.fg_data)

    def __getitem__(self,index):
        
        
        try:
            self.fg_data, self.fg_label, self.fg_mask = self._get_foreground_data()
            self.width = self.fg_data.shape[0]
            self.height = self.fg_data.shape[1]
        
            img_name, label_name  = self.bg_data[index]
            bg = self.crop_corner(img_name)
            if len(bg.shape) != 3:
                return self.__getitem__(int(np.random.choice(self.__len__(),1)))

            img = self._get_training_pair(bg)
            # img = copy.deepcopy(self.fg_data)
        except:
            return self.__getitem__(int(np.random.choice(self.__len__(),1)))
        label = copy.deepcopy(self.fg_label)    
        
        label[label>1] = 1
        if (np.max(label) - 1) < 1e-6:
            label = label * 255
        
        img = Image.fromarray(img.astype(np.uint8))
        label = Image.fromarray(label.astype(np.uint8))
        sample = {'image': img, 'label': label}

        #if self.split in ["train","full","shot"] :#== 'train' or self.split == 'full':
        return self.transform_train(sample)

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            st.RandomHorizontalFlip(),
            st.RandomScaleCrop(base_size=self.opt.val_size, crop_size=self.opt.crop_size, fill=0),
            st.RandomGaussianBlur(),
            st.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            st.ToTensor()])

        return composed_transforms(sample)

class SegmentationDataset(data.Dataset):
    def __init__(self, opt, split="train", folder=None, minnum=10):
        self.opt = opt
        self.split = split
        self.minnum=minnum
        self.test_dir = opt.data_dir if folder is None else folder
        data_dir = opt.data_dir if folder is None else folder

        portrait = BaseData(split=split, data_dir=data_dir, minnum=minnum)
        self.data = portrait.data
        print("split --> %s"%(split),len(self.data))
        if split == "train":
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label_name  = self.data[index]
        img_name = os.path.join(self.test_dir,img_name)
        label_name = os.path.join(self.test_dir,label_name)

        img = Image.open(img_name).convert('RGB')

        #if len(np.asarray(img).shape) == 2:
        img = np.asarray(img,dtype=np.uint8)
        if len(img.shape) == 2:
            img = np.stack((img,img,img),axis=2)
        if len(img.shape) == 4:
            img = img[:,:,:3]
        label = np.array(Image.open(label_name), dtype=np.uint8)
        label[label>1] = 1
        if (np.max(label) - 1) < 1e-6:
            label = label * 255
        if len(label.shape) == 4:
            label = label[:,:,3]
        if len(label.shape) != 2:
            label = label[:,:,0]

        img = Image.fromarray(img)
        label = Image.fromarray(label)
        sample = {'image': img, 'label': label}

        if self.split in ["train","full","shot"] :
            return self.transform_train(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_test(sample)


    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            st.RandomHorizontalFlip(),
            st.RandomScaleCrop(base_size=self.opt.val_size, crop_size=self.opt.crop_size, fill=0),
            st.RandomGaussianBlur(),
            st.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            st.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            st.FixScaleCrop(crop_size=self.opt.crop_size),
            st.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            st.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            st.FixedResize(size=self.opt.crop_size),
            st.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            st.ToTensor()])

        return composed_transforms(sample)

#######################  batch collate_fn  #########################

def segmentation_collate_fn(batch_data):
    
    imgs = [sample['image'] for sample in batch_data]
    labels = [sample['label'] for sample in batch_data]
    

    imgs = torch.stack(imgs)
    labels = torch.stack(labels)

    return imgs, labels

def scale_img_back(data,output_gpu=True,device=torch.device(0)):
    tmp = data.clone().permute(0,2,3,1)
    if output_gpu:
        for x in tmp:
            x *= torch.FloatTensor([0.229,0.224,0.225]).cuda(device=device)
            x += torch.FloatTensor([0.485,0.456,0.406]).cuda(device=device)
    else:
        for x in tmp:
            x *= torch.FloatTensor([0.229,0.224,0.225])
            x += torch.FloatTensor([0.485,0.456,0.406])

    return tmp.permute(0,3,1,2)



if __name__ == '__main__':
    
    opt = parse_opts()
    opt.base_size = 512
    opt.crop_size = 512
    
    import matplotlib.pyplot as plt
    portrait_train = SegmentationDataset(opt,split="train")

    dataloader = DataLoader(portrait_train,collate_fn=segmentation_collate_fn, batch_size=4, shuffle=True, pin_memory=True,num_workers=1)
    
    print(len(portrait_train))
    for ii, (imgs, labels) in enumerate(dataloader):
        
        
        img_grid = torchvision.utils.make_grid(scale_img_back(imgs,False), nrow=4)
        label_grid = torchvision.utils.make_grid(labels, nrow=4)

        torchvision.utils.save_image(img_grid, "img.png")
        torchvision.utils.save_image(label_grid, "label.png")
        
        break

