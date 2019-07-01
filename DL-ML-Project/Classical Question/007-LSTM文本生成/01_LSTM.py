"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

使用LSTM生成尼采的作品
"""

"""
Part 1. 数据准备
"""

# 下载语料并且将其转换为小写
import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

# 将字符序列向量化
maxlen = 60 # 提取60个字符组成的序列
step = 3 # 每3个字符采样一个新序列
sentences = [] # 保存所提取的序列
next_chars = [] # 保存目标（即下一个字符）

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text))) # 语料中唯一字符组成的列表
print('Unique characters:', len(chars))

# 字典，将唯一的字符映射为它在列表chars中的索引
char_indices = dict((char, chars.index(char)) for char in chars)

# 将字符one-hot编码为二进制数组
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

"""
Part 2. 构建网络

网络是单层的LSTM层，然后使用Dense分类器对所有可能的字符进行softmax
"""

from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


"""
Part 3. 训练语言模型并且从中采样

语言模型在采用时需要一定的随机性，因此引入softmax温度的参数，表示采样概率分布的熵
即下一个字符会多么的出人意料或者可预测
"""

# 给定目前已经生成的文本，从模型中预测下一个字符的概率
# preds是概率值组成的一维数组，这些概率值之和为1
# temperature是一个因子，用于定量描述输出分布的熵
# 更高的温度得到的是熵更大的采样分布，会生成更加出人意料的数据
# 更低的温度对应更小的随机性，以及更加可以预测的数据
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) # preds是原始分布重新加权后的结果
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 文本生成循环
# 反复训练并且生成文本，在每轮训练后都使用一系列不同的温度生成文本
import random
import sys

for epoch in range(1, 60):
    print('epoch', epoch)
    # 将模型在数据上拟合一次
    model.fit(x, y,
              batch_size=128,
              epochs=1)

    # 随机选择一个文本种子
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')
    # 尝试一系列不同的温度
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # 从种子文本开始，生成400个字符
        # 对目前生成的字符进行one-hot编码
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            # 对下一个字符进行采样
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
            sys.stdout.flush()

        print()

