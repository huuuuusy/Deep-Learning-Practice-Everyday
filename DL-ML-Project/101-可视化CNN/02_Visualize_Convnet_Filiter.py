""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
框架： TensorFlow-GPU == 1.13.1
      keras == 2.2.3
介绍： 可视化CNN的过滤器：
        理解CNN每个过滤器最容易接受的视觉模式或者视觉概念
      通过在输入空间进行梯度上升，可以观察CNN的过滤器：
        从空白输入图像开始，将梯度下降应用于卷积神经网络输入图像，目的是让某个滤波器的响应最大化
        得到的输入图像是选定过滤器具有最大响应的图像
      以在ImageNet上预训练的VGG16为例
"""

"""
Part 1. 下载模型
"""

from keras.applications import VGG16
from keras import backend as K 

import matplotlib.pyplot as plt 

model = VGG16(weights='imagenet', include_top=False)

"""
Part 2. 以block3_conv1层第０个过滤器为例
"""

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output 
loss = K.mean(layer_output[:, :, :, filter_index])

# 获取损失相对于输入的梯度
grads = K.gradients(loss, model.input)[0]
# 梯度标准化，+ 1e-5是防止除数为0
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate = K.function([model.input], [loss, grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# 以一张带有噪声的灰度图为例：
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# 进行梯度上升
step = 1.  # 每次梯度上升的步长
for i in range(40):
    # 计算损失值和梯度值
    loss_value, grads_value = iterate([input_img_data])
    # 沿着损失最大化的方向调节输入图像
    input_img_data += grads_value * step
def deprocess_image(x):
    # 对张量进行标准化，均值为0,标准差为1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # 将x控制在[0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # 转化为RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    # 构建一个损失函数，将第n个过滤器的激活最大化
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # 计算损失相对于输入图像的梯度
    grads = K.gradients(loss, model.input)[0]

    # 将梯度标准化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 返回给定图像的损失和梯度　
    iterate = K.function([model.input], [loss, grads])
    
    # 以带有噪声的灰度图为例
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # 运行40次梯度上升
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)
plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

"""
Part 3. 可视化所有卷积层的过滤器
"""

for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
    size = 64
    margin = 5

    # 空图像（黑色）
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):  # 遍历results的行
        for j in range(8):  # 遍历results的列
            # 生成layer_name层第i+(j*8)个过滤器的模式
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
            
            #plt.imshow(filter_img)
            #plt.show()

            # 将结果放到results网格第(i,j)个方块中
            
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # 显示results网格
    plt.figure(figsize=(20, 20))
    plt.imshow(results.astype(int)) # 需要将浮点数转化为整数，否则绘图时会发生错误
    plt.savefig("02_" + layer_name + ".png")
    plt.show()