""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
框架： TensorFlow-GPU == 1.13.1
      keras == 2.2.3
介绍： 可视化CNN的中间输出（中间激活）：理解CNN连续的层如何对输入进行变换
      对于给定输入，可视化网络中各个卷积层和池化层的特征图
"""

"""
Part 1. 下载猫狗数据集上的训练模型
"""

from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')
model.summary()

"""
Part 2. 显示待测试图片
"""

img_path = 'cat.1700.jpg'

from keras.preprocessing import image
import numpy as np 

# 将图片转换为一个4D张量
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

# 归一化处理
img_tensor /= 255

import matplotlib.pyplot as plt 

plt.imshow(img_tensor[0])
plt.show()

"""
Part 3. 构建每一层输入输出对应关系
"""

from keras import models

# 提取前8层的输出
layer_outputs = [layer.output for layer in model.layers[:8]]
# 创建模型，给定模型输入，可以返回输出:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# 返回8个Numpy数组组成的列表，每层激活对应一个Numpy数组
activations = activation_model.predict(img_tensor)

# 绘制第一层激活的通道输出，第一层是conv2d_5(Conv2D)(None, 148, 148, 32)，有32个输出通道
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
plt.show()

"""
Part 4. 将所有通道可视化
"""

import keras

# 存储每一层的名字
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    # 特征图中特征的个数
    n_features = layer_activation.shape[-1]

    # 特征图形状：(1, size, size, n_features)
    size = layer_activation.shape[1]

    # 平铺激活通道
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 将每个过滤器平铺到大的水平网格
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # 对特征进行后处理，为了显示的美观
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # 显示
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig("01_" + layer_name + ".png")
    
plt.show()