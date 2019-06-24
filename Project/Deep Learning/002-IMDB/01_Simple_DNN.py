 # @Author: huuuuusy
 # @GitHub: https://github.com/huuuuusy

 # IMDB来自电影数据库，包含50000条评论，其中25000条作为训练集，25000条作为测试集
 # 训练集和测试集均包含50%的正面评论和50%的负面评论
 # 本例使用最简单的神经网络对IMDB进行情感分析

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
    results = np.zeros(len(sequences), dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

