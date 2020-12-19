""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
框架： TensorFlow-GPU == 1.13.1
      keras == 2.2.3
介绍： 借助已经在大型数据集上训练良好的卷及神经网络，解决小型数据集无法充分学习的问题
      卷积神经网络主要由以下两大部分构成：
        1. 卷积基: 由一系列池化层和卷积层构成
        2. 密集连接分类器: 针对分类任务设计的由一系列全连接网络构成的分类器
      特征提取就是取出之前训练好的网络的卷积基，在上面运行新的数据，然后训练一个新的密集连接分类器
      使用VGG16模型作为卷积基进行研究
"""

"""
Part 1. 下载VGG16模型
"""

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', # 使用在imagenet上训练的VGG16的卷积基
                  include_top=False,
                  input_shape=(150, 150, 3)) # 不包含密集连接分类器
                
conv_base.summary()

"""
Part 2. 快速特征提取

在VGG16的卷积基结构上添加密集连接分类器，以完成猫和狗的分类任务，有如下两种方式可以选择：

1.不使用数据增强的快速特征提取
在数据集上运行卷积基，将输出保存成numpy数组，然后将这个数组作为输入，训练独立的密集连接分类器中进行训练。
优点: 速度快，计算代价低，在每张图上只运行一次卷积基
缺点: 因为每张图只运行一次卷积基，所以无法进行数据增强

2.使用数据增强的特征提取
扩展已有的卷积基模型，在顶部添加密集连接分类器，然后在训练数据上端到端地运行整个模型。
优点:　可以使用数据增强
缺点: 卷积基参与大量的运算，运算代价高

此处使用第一种快速特征提取
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/hu/Downloads/dogs-vs-cats_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255) # 生成器不使用数据增强，只进行归一化
batch_size = 20

def extract_features(dictionary, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512)) # 储存从VGG16的卷积基输出的特征
    labels = np.zeros(shape=(sample_count)) # 储存标签
    generator = datagen.flow_from_directory(dictionary,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        # 将输入通过conv_base进行计算，最终从VGG16输出的是4, 4, 512的张量(参考VGG16的结构)
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >= sample_count:
            break # 因为生成器在循环时不断生成数据，所以必须在读完图像后终止循环
    
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

"""
Part 3. 构建密集连接分类器
"""

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5)) # 使用dropout预防过拟合
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels, # 密集连接分类器的输入是VGG16卷积基的输出
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

"""
Part 4. 绘制图像
"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("03_VGG16_Fast_Feature_Extraction_Accuracy.png")
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("03_VGG16_Fast_Feature_Extraction_Loss.png")
plt.show()