# -*- coding:utf-8 -*-
"""
RMB分类
"""
import os, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np

from dataset import RMBDataset
from utils import split_dataset
from lenet import LeNet

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_lenet(split_dir):
    """训练lenet"""
    set_seed()  # 设置随机种子
    rmb_label = {"1": 0, "100": 1}

    # 参数设置
    MAX_EPOCH = 10
    BATCH_SIZE = 16
    LR = 0.01
    log_interval = 10
    val_interval = 1

    """Step 1: 数据读取"""
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # 对训练数据进行变换,添加RandomCrop进行数据增强
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 对验证数据进行变换
    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建RMBDataset实例
    train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
    valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

    # 构建DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    """Step 2: 模型"""
    net = LeNet(classes=2)
    net.initialize_weights()

    """Step 3: 损失函数"""
    criterion = nn.CrossEntropyLoss()  

    """Step 4: 优化器"""
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)   

    """Step 5: 训练"""  
    train_curve = []
    valid_curve = []

    figure_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result')

    for epoch in range(MAX_EPOCH):
        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().numpy()

                    loss_val += loss.item()

                valid_curve.append(loss_val/valid_loader.__len__())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, correct_val / total_val))
    
    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    figure_path = os.path.join(figure_dir, '0201.png')
    plt.savefig(figure_path)
    plt.close()

            
if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'RMB_data')
    split_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'RMB_split')
    
    #split_dataset(dataset_dir, split_dir)
    train_lenet(split_dir)

