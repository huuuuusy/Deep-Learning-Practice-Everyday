"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

MNIST是深度学习领域最常用的数据集，其中包含60000张训练图像和
10000张测试图像。数据集是针对手写数字的分类，每张图是28×28的
灰度图
本例使用最简单的卷积网络对MNIST进行分类

"""
"""
Part 1. 数据预处理，和上个模型相同
"""
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

"""
Part 2. 构建Simple CNN模型
"""
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28,  1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test loss:', test_loss)
print('test accuracy:', test_acc)