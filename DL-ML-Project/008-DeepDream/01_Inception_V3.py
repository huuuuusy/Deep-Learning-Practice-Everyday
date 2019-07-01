"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

利用ImageNet上的预训练网络，实现DeepDream效果
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

