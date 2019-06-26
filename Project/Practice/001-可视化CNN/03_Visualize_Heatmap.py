""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

可视化图像中类激活的热力图：理解图像中哪部分被识别为属于某种类别，从而定位图像中的物体

类激活可视化是对输入图像生成类激活热力图
类激活热力图是与特定输出类别相关的二维分数网格，对任何输入图像的每个位置都要进行计算，表示每个位置对该类别的重要程度

具体实现方式参考：
Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization
"""

"""
Part 1. 下载模型
"""

from keras.applications.vgg16 import VGG16
from keras import backend as K 

K.clear_session()

# 下载包含密集连接分类器的VGG16网络权重
model = VGG16(weights='imagenet')

"""
Part 2. 下载图片
"""

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = '/home/hu/Documents/Huuuuusy-Github/Deep-Learning-Practice-Everyday/Github/Project/Practice/001-可视化CNN/creative_commons_elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))

# `x` 是形状为(224, 224, 3)的float32格式的numpy数组
x = image.img_to_array(img)

# 添加一个维度后，将数组转化为(1, 224, 224, 3)格式的批量
x = np.expand_dims(x, axis=0)

# 对批量进行预处理（按通道进行颜色标准化）
x = preprocess_input(x)

# 对图像进行预测，并打印在1000个分类中最有可能的三种分类以及概率
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

"""
Part 3. Grad-CAM算法
"""

import matplotlib.pyplot as plt 

# 预测向量中的非洲象元素
african_elephant_output = model.output[:, 386]

# block5_conv3层的输出特征图，这是VGG16的最后一个卷积层
last_conv_layer = model.get_layer('block5_conv3')

# 非洲象类别对于block5_conv3输出特征的梯度
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# 形状为(512,)的向量，每个元素是特定特征图通道的梯度平均大小
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# 访问刚刚定义的量：对于给定的样本图像，pooled_grads和block5_conv3层的输出特征图
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

# 将特征图组数的每个通道乘以该通道对于大象类别的重要程度
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# 得到的特征图的逐通道平均值即为类激活的热力图
heatmap = np.mean(conv_layer_output_value, axis=-1)
# 将热力图标准化到0~1范围内
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.savefig("03_heatmap.png")
plt.show()

"""
Part 4. 叠加热力图
"""

from cv2 import cv2 as cv2

img = cv2.imread(img_path)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)

# 将热力图添加到原始图像上
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 是热力图的强度因子
superimposed_img = heatmap * 0.4 + img

cv2.imwrite('03_result.png', superimposed_img)