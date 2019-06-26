"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

IMDB来自电影数据库，包含50000条评论，其中25000条作为训练集，25000条作为测试集
训练集和测试集均包含50%的正面评论和50%的负面评论
本例使用最简单的神经网络对IMDB进行情感分析
"""

# 下载数据
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print('the first review: ', x_train[0])
print('the first train label: ', y_train[0])

# 构建word_index，将单词映射为整数索引的字典
word_index  = imdb.get_word_index()
print('the length of word index: ', len(word_index))
# print(word_index)

# 将word_index颠倒，构建reverse_word_index,即将索引转换为单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 解码第一条评论,将索引值减少3,因为前三个数字默认为'padding', 'start of sequence', 'unknown'的保留索引值
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in x_train[0]])
print('the decoded first review: ', decoded_review)

import numpy as np

def vectorize_sequences(sequences, dimension = 10000):
    # 先构造一个全0矩阵
    results = np.zeros((len(sequences), dimension))
    # 然后将指定位置的元素置1
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 将数据向量化
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)
print('train data shape: ', x_train.shape)
print('test data shape: ', x_test.shape)
print('the first vectorize review in train dataset: \n', x_train[0])
print('the first vectorize review in test dataset: \n', x_test[0])

# 将标签向量化
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

# 构建神经网络
from keras import models
from keras import layers

simple_dnn = models.Sequential()
simple_dnn.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
simple_dnn.add(layers.Dense(16, activation='relu'))
simple_dnn.add(layers.Dense(1, activation='sigmoid'))

# 自定义损失函数和指标
from keras import losses
from keras import metrics
from keras import optimizers

simple_dnn.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

# 测试神经网络
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = simple_dnn.fit(partial_x_train,
                        partial_y_train,
                        epochs=4,
                        batch_size=512,
                        validation_data=(x_val, y_val))

# 绘制图像
# 训练损失和验证损失
import matplotlib.pyplot as plt

val_loss = history.history['val_loss']
val_binary_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
binary_accuracy = history.history['binary_accuracy']

epochs = range(1, len(binary_accuracy)+1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 训练准确率和验证准确率
plt.clf()   # clear figure

plt.plot(epochs, binary_accuracy, 'bo', label='Training acc')
plt.plot(epochs, val_binary_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 测试集准确率
results = simple_dnn.evaluate(x_test, y_test)
print('evaluate test result:', results)