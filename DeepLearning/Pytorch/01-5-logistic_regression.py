# -*- coding:utf-8 -*-
"""
逻辑回归
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

class LR(nn.Module):
    """自定义逻辑回归的类"""
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

def logistic_regression():
    """Step 1: 生成数据"""
    sample_nums = 100
    mean_value = 1.7
    bias = 1
    n_data = torch.ones(sample_nums, 2)
    x0 = torch.normal(mean_value * n_data, 1) + bias # 类别0数据
    y0 = torch.zeros(sample_nums) # 类别0标签
    x1 = torch.normal(-mean_value * n_data, 1) + bias # 类别1数据
    y1 = torch.ones(sample_nums) # 类别1标签
    train_x = torch.cat((x0, x1), 0) # 训练数据
    train_y = torch.cat((y0, y1), 0) # 训练标签
    print('train_x: {}\ntrain_y: {}'.format(train_x, train_y))

    """Step 2: 选择模型"""
    lr_net = LR() # 实例化逻辑回归模型

    """Step 3: 选择损失函数"""
    loss_fn = nn.BCELoss()

    """Step 4: 选择优化器"""
    lr = 0.01
    optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

    """Step 5: 模型训练"""
    figure_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result')
    for iteration in range(1000):
        y_pred = lr_net(train_x) # 前向传播
        loss = loss_fn(y_pred.squeeze(), train_y) # 计算loss
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        optimizer.zero_grad() # 清空梯度

        if iteration % 20 == 0:

            mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
            correct = (mask == train_y).sum()  # 计算正确预测的样本个数
            acc = correct.item() / train_y.size(0)  # 计算分类准确率

            print('iteration {}: acc={}'.format(iteration, acc))

            plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
            plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

            w0, w1 = lr_net.features.weight[0]
            w0, w1 = float(w0.item()), float(w1.item())
            plot_b = float(lr_net.features.bias[0].item())
            plot_x = np.arange(-6, 6, 0.1)
            plot_y = (-w0 * plot_x - plot_b) / w1

            plt.xlim(-5, 7)
            plt.ylim(-7, 7)
            plt.plot(plot_x, plot_y)

            plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
            plt.legend()

            figure_path = os.path.join(figure_dir, '0105_iteration_{}.png'.format(iteration))
            plt.savefig(figure_path)
            plt.close()

            if acc > 0.99:
                break

if __name__ == "__main__":
    logistic_regression()