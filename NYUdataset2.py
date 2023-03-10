import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset


def list_dir(root, extname='.npy'):
    extlen = len(extname)
    files = os.listdir(root)
    filelist = []
    for i in files:
        # print(i[(-1)*extlen:])
        if i[(-1)*extlen:] == extname:
            # print('true')
            filelist.append(i)
    filelist.sort(key=lambda x: x.split('.')[0])
    # print(filelist)
    return filelist
    
class NYUv2Dataset(Dataset):
    def __init__(self, dataroot='/home/dyf/database/NYUv2/aftercrop/', phase='train', resolution=(224,224), random_crop=False, artifical_mask=False, extname='.npy', exp=10.1):
        self.dataroot = dataroot
        self.phase = phase
        self.extname = extname
        self.exp = exp
        self.files = list_dir(dataroot + phase + '/gt', extname=extname)
        # print(self.files)
        self.H = resolution[0]
        self.W = resolution[1]
        self.random_crop = random_crop
        self.artifical_mask = artifical_mask
        self.lendata = len(self.files)

    def __len__(self):
        return self.lendata

    def read_data_fix(self, index):
        folder = self.dataroot + self.phase
        # print(self.files)
        if self.extname == '.npy':
            raw = np.load(folder + '/raw/' + self.files[index])
            gt = np.load(folder + '/gt/' + self.files[index])
            rgb = np.load(folder + '/rgb/' + self.files[index])
        else:
            raw = cv2.imread(folder + '/raw/' + self.files[index], cv2.IMREAD_UNCHANGED)
            gt = cv2.imread(folder + '/gt/' + self.files[index], cv2.IMREAD_UNCHANGED)
            rgb = cv2.imread(folder + '/rgb/' + self.files[index], cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = rgb.transpose(2, 0, 1)

        h, w = rgb.shape[1], rgb.shape[2]
        # print(h, w, self.H, self.W)
        # raise ValueError('stop')
        assert h == gt.shape[0] 
        assert w == gt.shape[1]
        assert h == raw.shape[0] 
        assert w == raw.shape[1]

        # 若大于原图， 自动采用原图分辨率
        H = self.H if self.H<=h else h
        W = self.W if self.W<=w else w

        if self.random_crop:
            sh = random.randint(0, h-H)
            sw = random.randint(0, w-W)
        else:
            sh = int(round(h-H) / 2)
            sw = int(round(w-W) / 2)  
        
        rgb = rgb[:, sh:sh+H, sw:sw+W]
        gt = gt[sh:sh+H, sw:sw+W]
        raw = raw[sh:sh+H, sw:sw+W]

        return gt, raw, rgb, self.files[index]

    def read_data_artimask(self, index):
        folder = self.dataroot + self.phase

        if self.extname == '.npy':
            gt = np.load(folder + '/gt/' + self.files[index])
            rgb = np.load(folder + '/rgb/' + self.files[index])
        else:
            gt = cv2.imread(folder + '/gt/' + self.files[index], cv2.IMREAD_UNCHANGED)
            rgb = cv2.imread(folder + '/rgb/' + self.files[index], cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = rgb.transpose(2, 0, 1)


        h, w = rgb.shape[1], rgb.shape[2]
        # print(h, w)
        assert h == gt.shape[0] 
        assert w == gt.shape[1]
        H = self.H if self.H<h else h
        W = self.W if self.W<w else w

        if self.random_crop:
            sh = random.randint(0, h-H)
            sw = random.randint(0, w-W)
        else:
            sh = int(round(h-H) / 2)
            sw = int(round(w-W) / 2)  

        rgb = rgb[:, sh:sh+H, sw:sw+W]
        gt = gt[sh:sh+H, sw:sw+W]
        raw = gt.copy()

        num_mask = 4
        for i in range(num_mask):
            mask_size = [random.randint(int(H/6), int(H/4)), random.randint(int(W/6), int(W/4))]
            mask_point = [random.randint(0, H-mask_size[0]), random.randint(0, H-mask_size[1])]
            raw[mask_point[0]:(mask_point[0]+mask_size[0]), mask_point[1]:(mask_point[1]+mask_size[1])] = 0
        raw = np.array(raw, dtype=np.float32)

        return gt, raw, rgb, self.files[index]

    def __getitem__(self, index):
        # assert index <= len(self), 'index range error'
        if self.artifical_mask:
            gt, raw, rgb, name = self.read_data_artimask(index)
        else:
            gt, raw, rgb, name = self.read_data_fix(index)
        gt = np.float32(gt)
        raw = np.float32(raw)
        gt = gt/self.exp
        raw = raw/self.exp
        gt = torch.from_numpy(gt)
        raw = torch.from_numpy(raw)
        rgb = torch.from_numpy(np.float32(rgb))/256

        return gt, raw, rgb, name

if __name__=='__main__':
    dataset = NYUv2Dataset(dataroot='/home/dyf/database/SunRGBD/SUNRGBD/data/', phase='train', resolution=(384,384), random_crop=True, artifical_mask=False, extname='.png')
    len = dataset.lendata
    print(len)
    for i in range (len):
        _, raw, _, _ = dataset.__getitem__(i)
        print(raw.max())
