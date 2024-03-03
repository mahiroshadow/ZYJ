import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class MEGC2019(Dataset):
    """MEGC2019 dataset class with 3 categories"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList, 'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]), 'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label[idx]

    def __len__(self):
        return len(self.imgPath)


class MEGC2019_SI_MeRoI(torch.utils.data.Dataset):  # Flow +  Flow
    """MEGC2019_SI dataset class with 3 categories and other side information"""

    def __init__(self, imgList, transform=None,transform1=None):
        self.imgPath = []
        self.imgPath1 = []
        self.label = []
        self.dbtype = []
        with open(imgList, 'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.imgPath1.append(texts[0].replace('Flow', 'Flow_puzzle'))
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform
        self.transform1=transform1

    def __getitem__(self, idx):
        print()
        img = Image.open("".join(self.imgPath[idx]), 'r').convert('RGB')
        img1 = Image.open("".join(self.imgPath1[idx]), 'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)
        return {"data": img, "data1": img1, "class_label": self.label[idx], 'db_label': self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)


class MEGC2019_SI_MeRoI_single(torch.utils.data.Dataset):  # Flow
    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList, 'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]), 'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return {"data": img, "class_label": self.label[idx], 'db_label': self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)

class IRNN2023(Dataset):
    def __init__(self,pth,transform):
        super(IRNN2023,self).__init__()
        self.df=pd.read_csv(pth)
        self.transform=transform

    def __getitem__(self, index):
        image_seq=[]
        final_img=None
        for (root,dir,files) in os.walk(self.df["pth"][index]):
            for (idx,file) in enumerate(files):
                img=self.transform(Image.open(root+"\\"+file))
                if idx<=4:
                    image_seq.append(img.detach().numpy())
                else:
                    final_img=np.array(img)
        image_seq=np.array(image_seq)
        return image_seq,self.df["class_label"][index],final_img

    def __len__(self):
        return len(self.df)

