"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

使用Embedding层作为模型第一层，在完成主任务的同时进行词嵌入学习
"""

"""
Part 1. 数据预处理
"""

from keras.datasets import imdb
from keras import preprocessing

max_feature = 10000 # 作为特征的单词个数，不属于这些单词的其他词汇被认为是非常见词
maxlen = 100 # 在maxlen个单词后截断文本，这些单词属于前max_feature个常见单词

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)

# 将数据列表转换为形状为(samples, maxlen)的二维张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

"""
Part 2. 构建模型，使用Embedding层 
"""

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()

# Embedding层只能作为模型的第一层，将正整数转化为具有固定大小的向量 
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.summary()

history = model.fit(x_train, y_train,  epochs=10, batch_size=32, validation_split=0.2)