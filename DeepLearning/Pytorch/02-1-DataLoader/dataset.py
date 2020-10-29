# -*- coding:utf-8 -*-
"""
RMBDataset类的定义
"""
import os
import random
from torch.utils.data import Dataset
import glob
from PIL import Image

random.seed(1)
rmb_label = {"1": 0, "100": 1}

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform


    def __getitem__(self, index):
        img_path, label = self.data_info[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    

    def __len__(self):
        return len(self.data_info)

    
    def get_img_info(self, data_dir):
        data_info = []
        for root, dirs, files in os.walk(data_dir):
            for sub_dir in dirs:
                imgs = glob.glob(os.path.join(root, sub_dir, '*.jpg'))

                for i in range(len(imgs)):
                    img_path = imgs[i]
                    label = self.label_name[sub_dir]
                    data_info.append((img_path, int(label)))
        return data_info

                    

    


    

