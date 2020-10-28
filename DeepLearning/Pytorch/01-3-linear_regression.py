# -*- coding:utf-8 -*-
"""
线性回归
"""
import torch
import matplotlib.pyplot as plt
import os

torch.manual_seed(10)

def linear_regression():
    lr = 0.05

    # train data
    x = torch.rand(20,1)*10
    y = 2*x + (5+torch.randn(20,1))
    print('x: {}\ny: {}'.format(x,y))

    # parameter
    w = torch.randn((1), requires_grad=True)
    b = torch.zeros((1), requires_grad=True)

    figure_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result')
    for iteration in range(1000):
        wx = torch.mul(w, x)
        y_pred = torch.add(wx, b)

        # MSE Loss
        loss = (0.5*(y-y_pred)**2).mean()

        # backward
        loss.backward()

        # update
        w.data.sub_(lr*w.grad)
        b.data.sub_(lr*b.grad)
        
        # 清零张量梯度(重要)
        w.grad.zero_()
        b.grad.zero_()

        print('iteration {}: loss={}'.format(iteration, loss.data.numpy()))

        # draw plot
        if iteration % 20 == 0:
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
            plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
            plt.xlim(1.5, 10)
            plt.ylim(8, 28)
            plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
            figure_path = os.path.join(figure_dir, '0103_iteration_{}.png'.format(iteration))
            plt.savefig(figure_path)
            plt.close()

            if loss.data.numpy() < 1:
                break

if __name__ == "__main__":
    linear_regression()
    