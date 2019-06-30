"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

使用其他模型上预训练的词嵌入，然后加载到模型中
"""

"""
Part 1. 数据预处理
访问http://ai.stanford.edu/~amaas/data/sentiment/ 并且下载原始的IMDB数据集
解压数据集，并将训练评论转化为字符串列表，每个字符串对应一行评论
"""

import os

imdb_dir = '/home/hu/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

"""
Part 2. 对数据进行分词
"""                

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100  # 从100词后截断数据
training_samples = 10000  # 使用10000个数据训练(小样本)
validation_samples = 10000  # 在10000个数据上验证
max_words = 10000  # 只考虑数据集前10000个最常见的单词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将数据划分为训练集和验证集，但是首先需要打乱数据，因为一开始样本是排好序列的（先是负面评论，再是正面评论）
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

"""
Part 3. 下载GloVe词嵌入
打开https://nlp.stanford.edu/projects/glove/ 并且下载glove.6B.zip, 然后解压
"""  

# 对嵌入进行预处理
# 对解压后的文件进行解析，构建将单词（字符串）映射为向量表示（数值向量）的索引
glove_dir = '/home/hu/Downloads/glove.6B/'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

"""
Part 4. 准备GloVe词嵌入矩阵
构建形状为(max_words, embedding_dim）的矩阵，以确保可以加入到Embedding层中
对于单词索引为i的单词，词嵌入矩阵的元素i即为该单词对应的embedding_dim维向量
索引0不代表任何标记，只是一个占位符
""" 

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # 如果索引找不到对应的词，嵌入词向量将全为0
            embedding_matrix[i] = embedding_vector

"""
Part 5. 模型
""" 

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 加载GloVe嵌入
# 将准备好的GloVe矩阵加入到Embedding层中，即模型的第一层。此外需要将该层冻结，以避免破坏预训练部分所保存的信息
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')


"""
Part 6. 绘制图像
""" 

import matplotlib.pyplot as plt 

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("05_Pretrained_Embedding_Accuracy.png")
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("05_Pretrained_Embedding_Loss.png")
plt.show()
