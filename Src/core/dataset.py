import sys
import os
import numpy as np
import imageio
import cv2 as cv
import torch
from torch import Tensor
from torch.utils.data import Dataset
sys.path.append(r'../')
from path import PATH_TRN_VAL, DIR_TRN_VAL

mean_RGB = (0.6437256496846695, 0.5872457606424337, 0.5612083244276778)
var_RGB = (0.08254510705692104, 0.0884365306266636, 0.0887025657385277)
std_RGB = [np.sqrt(var_) for var_ in var_RGB]

def rdrImgDft(path):
    img = imageio.imread(path)
    if img.ndim<3:
        img = np.stack([img]*3, 2)
    return img
def rdrAnnotDft(path):
    with open(path, 'r', encoding='UTF-8') as f:
        lines=f.readlines()
    return np.asarray([line[:-1] for line in lines]), [line[-1]  for line in lines]
def ldrDft(img,):
    '''(H,W,C)->(C,H,W)
       [0,255]->[0,1]  '''
    img=img.transpose(2,0,1)/255
    img=img
    return Tensor(img)

def ldrWht(img,):
    ''' (pixel-mean_ch)/std_ch '''
    return (ldrDft(img)-mean_RGB)/std_RGB

class OCR_DATASET(Dataset):
    def __init__(self, path=PATH_TRN_VAL, folders=DIR_TRN_VAL,
                 readerImg=rdrImgDft, readerAnnot=rdrAnnotDft,
                 transforms=None, loader=ldrDft):
        self.readerImg = readerImg
        self.readerAnnot = readerAnnot
        self.transforms=transforms
        self.loader=loader
        if len(folders)>1:
            self.annot = True
            dirImg, dirLbl = folders
            pathImg = os.path.join(path,dirImg)
            pathLbl = os.path.join(path,dirLbl)
        else:
            self.annot = False
            dirImg = folders
            pathImg = os.path.join(path,dirImg)
        lstImg = os.listdir(pathImg)
        if self.annot:
            lstLbl = [os.path.splitext(img)[0]+'.txt' for img in lstImg]
            self.paths = [(os.path.join(pathImg, Img),
                           os.path.join(pathLbl, Lbl))
                           for Img, Lbl in zip(lstImg, lstLbl)]
        else:
            self.paths = [os.path.join(path, Img) for Img in lstImg]
    def __getitem__(self, idx):
        paths = self.paths[idx]
        if self.annot:
            img = self.readerImg(paths[0])
            loc,txt = self.readerAnnot(paths[0])
        else:
            img = self.readerImg(paths[0])

        if isinstance(img, type(None)):
            print(paths)
        # transforms
        if not isinstance(self.transforms, type(None)):
            if self.annot:
                img = self.transforms(img, loc)
            else:
                img = self.transforms(img)
        # loader
        if not isinstance(self.loader, type(None)):
            img = self.loader(img)

        ret = []
        ret.append(img)
        if self.annot:
            return img, loc, txt
        else:
            return img

    def __len__(self,):
        return len(self.paths)


if __name__=='__main__':
    dataset = OCR_DATASET()
    for ith,(img, lbl) in enumerate(dataset):
