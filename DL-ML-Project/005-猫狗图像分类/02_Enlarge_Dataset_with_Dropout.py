""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
框架： TensorFlow-GPU == 1.13.1
      keras == 2.2.3
介绍： 使用图像增强技术，扩大小型数据集中的样本数量
      使用dropout正则化，减缓过拟合
"""

"""
Part 1. 图像随机增强
"""

import os, shutil

base_dir = '/home/hu/Downloads/dogs-vs-cats_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
                        
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=40, # 随机旋转的角度
                             width_shift_range=0.2, # 水平移动的比例
                             height_shift_range=0.2, # 垂直移动的比例
                             shear_range=0.2, # 随机错切变换角度
                             zoom_range=0.2, # 随机缩放范围
                             horizontal_flip=True, # 随机将一半图像水平翻转
                             fill_mode='nearest')

from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]   

# 选择一张图进行增强
img_path = fnames[3]
# 调整大小
img = image.load_img(img_path, target_size=(150, 150))
# 将图像变为数组， 数组形状为(150, 150, 3)
x = image.img_to_array(img)
# 将形状变为(1, 150, 150, 3)
x = x.reshape((1,) + x.shape)
# 生成随机变换的图像批次, 随机变化4次
import matplotlib.pyplot as plt

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

"""
Part 2. 构建随机增强后的数据
"""

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# 注意测试集的数据不进行增强
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # 目标文件夹
        train_dir,
        # 所有数据都将转化为150 * 150
        target_size=(150, 150),
        batch_size=32,
        # 使用binary标签，因为损失函数使用的是binary_crossentropy
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

"""
Part 3. CNN添加Dropout
"""

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

from keras import optimizers
from keras import losses
model.compile(loss=losses.binary_crossentropy,
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

"""
Part 4. 模型训练及保存
"""

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

model.save('cats_and_dogs_small_2.h5')

"""
Part 5. 绘制图像
"""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("02_Enlarge_Dataset_with_Dropout_Accuracy.png")
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("02_Enlarge_Dataset_with_Dropout_Loss.png")
plt.show()