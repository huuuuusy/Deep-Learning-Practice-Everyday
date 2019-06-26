""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

微调是指将其顶部的几层解冻，并将解冻的层和新增加的部分进行联合训练。具体步骤：
1. 在已经训练好的卷积基上添加自定义网络
2. 冻结基网络
3. 训练所添加的部分
4. 解冻基网络的部分层
5. 联合训练解冻层和添加的自定义网络

Part 1 ～ Part 3 和04_VGG16_End2ENd_Feature_Extraction一致
Part 4解冻部分卷积基的卷积层，进行微调
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
Part 2. 使用数据增强
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/hu/Downloads/dogs-vs-cats_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

from keras.preprocessing.image import ImageDataGenerator

# 对训练集进行数据增强
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# 注意：对测试集不能进行数据增强！！！
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        # 因为使用binary_crossentropy作为损失函数，所以要使用binary的标签
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

"""
Part 3. 构建完整的网络结构
卷积基是VGG16的卷积基
分类器是针对猫狗问题设计的二分类密集连接分类器
"""

from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

"""
Part 4. 微调模型

参考上面的VGG16卷积基，将从block5_conv1层进行解冻，之前的层继续冻结，之后的层将参与训练

微调顶部卷积层的原因：
1. 卷积基中更靠近底部的卷积层编码的是更加通用的可复用特征，而更靠近顶部的层编码的是更专业化的特征。微调顶部将帮助模型更适应特定的问题
2. 微调的层越多，训练的参数将越多，最终过拟合的风险越大。所以最好的选择是微调卷积基靠近顶部的网络层
"""

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), # 适当降低学习率，以保证微调的效果
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

model.save('cats_and_dogs_small_4.h5')

"""
Part 5. 绘制图像，使用平滑函数
"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""
Part 6. 测试
"""

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)