"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
框架： TensorFlow-GPU == 1.13.1
      keras == 2.2.3
介绍： 利用ImageNet上的预训练网络，实现DeepDream效果
"""

"""
Part 1. 模型准备

加载预训练的Inception V3模型
"""

from keras.applications import inception_v3
from keras import backend as K 

# 无需训练模型，所以设定参数禁止所有训练过程
K.set_learning_phase(0)

# 构建不包含全连接层的InceptionV3网络
# 使用预训练权重加载模型
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# 字典将层的名称映射为系数
# 系数定量表示该层激活对所需最大化损失的贡献大小
layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}

"""
Part 2. 定义需要最大化的损失
"""

# 创建字典，将层名称映射为层的实例
layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output # 获取层的输出

    # 将该层特征的L2范数添加到loss中
    # 为了避免出现边界伪影，损失仅包含非边界像素
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

"""
Part 3. 梯度上升
"""

# 存储梦境图像
dream = model.input

# 计算损失相对于梦境图像的梯度
grads = K.gradients(loss, dream)[0]

# 将梯度标准化
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# 给定一张输出图像，设置keras函数来获取损失值和梯度值
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# 函数运行iterations次梯度上升
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

"""
Part 4. 辅助函数
"""
import scipy
from keras.preprocessing import image

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


def preprocess_image(image_path):
    # 打开图像、改变图像大小、将图像格式转换为Inception V3可处理的张量
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # 将张量转换为有效图像
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

"""
Part ５. 在多个连续尺度上运行梯度上升
"""

import numpy as np
step = 0.01  # 梯度上升步长
num_octave = 3  # 运行梯度上升的尺度个数
octave_scale = 1.4  # 两个尺度之间的大小比例
iterations = 20  # 在每个尺度上运行梯度上升的步数

# 如果损失大于10，将停止梯度上升过程
max_loss = 10.

base_image_path = '001.jpg'

img = preprocess_image(base_image_path)

# 准备一个由形状元组组成的列表，其定义运行梯度上升的不同尺度
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

# 反转列表为升序
successive_shapes = successive_shapes[::-1]

# 将图像的数组缩放到最小尺寸
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

save_img(img, fname='final_dream.png')

