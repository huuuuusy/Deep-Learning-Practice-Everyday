""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
框架： TensorFlow-GPU == 1.13.1
      keras == 2.2.3
介绍： Reuters数据集是新闻的多分类文本数据集，有46个主题
      每个主题的数量不同，但是每个主题的训练集中至少有10个样本
"""

# 下载数据
from keras.datasets import reuters
# 取出现频率最高的10000词构成词典，将数据中的单词用索引值表示
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
print(len(x_train))
print(len(x_test))
print('the first data:\n', x_train[0])

# 将索引解析为单词
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in x_train[0]])
print('the first decode data:\n', decoded_review)

# 准备数据
import numpy as np 

# 对训练数据和测试数据进行one-hot编码
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

# 对labels进行one-hot编码
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1. 
    return results

y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

print('train data shape: ', x_train.shape)
print('test data shape: ', x_test.shape)
print('train label shape: ', y_train.shape)
print('test label shape: ', y_test.shape)

# 构建神经网络

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 测试神经网络
# 取1000个样本作为训练时的验证集，剩下的作为训练集
x_val = x_train[:1000]
x_train = x_train[1000:]
y_val = y_train[:1000]
y_train = y_train[1000:]

history = model.fit(x_train, y_train,
                    epochs=8,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 绘图 
# 训练损失和验证损失
import matplotlib.pyplot as plt 

val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
loss = history.history['loss']
acc = history.history['acc']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo',  label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation liss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("01_Simple_DNN_Loss.png")
plt.show()

# 训练准确率和测试准确率
plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("01_Simple_DNN_Accuracy.png")
plt.show()

# 测试集准确率
results = model.evaluate(x_test, y_test)
print('evaluate test result:', results)