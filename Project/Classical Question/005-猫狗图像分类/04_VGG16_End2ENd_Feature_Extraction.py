""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

使用数据增强的特征提取：
扩展已有的卷积基模型，在顶部添加密集连接分类器，然后在训练数据上端到端地运行整个模型。
优点:　可以使用数据增强
缺点: 卷积基参与大量的运算，运算代价高
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
Part 2. 使用数据增强的特征提取
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
Part 4. 冻结卷积基并训练
VGG16的参数非常多，因此需要冻结卷积基
"""

# 在未冻结之前，有30个weights需要被训练:
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

# 冻结的方式是将trainable参数设置为false，冻结后只有4个weights需要被训练：
conv_base.trainable = False
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

# 训练
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

model.save('cats_and_dogs_small_3.h5')

"""
Part 5. 绘制图像
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

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()