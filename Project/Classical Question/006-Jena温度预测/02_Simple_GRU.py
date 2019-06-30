"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

使用两层GRU + Dropout进行训练
GRU是门控循环单元，与LSTM原理基本相同，但是有一些简化，因此运行的计算代价更低
"""

"""
Part 1. 数据准备
http://www.bgc-jena.mpg.de/wetter/
"""

import os

data_dir = '/home/hu/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

# 读取数据
f = open(fname)
data = f.read()
f.close()

# 处理数据
# headers是数据的第一行，是每列数据的名字，需要单独提取 
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(lines[0])
print(len(lines))

"""
Part 2. 解析数据
"""

import numpy as np 

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
    
print(float_data.shape)
print(float_data[0])

# 绘制气温的变化图，可以看出气温随时间变化的周期性
from matplotlib import pyplot as plt

temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)
plt.show()

"""
Part 3. 处理数据

lookback = 720表示给定过去720个时间步的数据
每个时间步是10分钟，720个时间步是过去5天的数据

steps = 6表示每6个时间步做一次采样
每一个小时记录一个数据点

delay = 144表示预测接下来24个小时之后的数据

需要对数据进行如下两部分操作:
1. 将数据标准化，在相似的范围内取较小的值
2. 编写python生成器，以当前的浮点数作为输入，并从最近的数据中生成数据批量，同时生成未来的目标温度
（因为样本数据高度冗余，即相邻的样本大部分数值相同，所以显式保存数据会浪费；应该利用原始数据即时生成样本）
"""

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

# 准备训练生成器，验证生成器，测试生成器

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# 为了查看整个验证集，需要从val_gen中抽取多少次
val_steps = (300000 - 200001 - lookback) // batch_size

# 为了查看整个测试集，需要从test_gen中抽取多少次
test_steps = (len(float_data) - 300001 - lookback) // batch_size

"""
Part 4. GRU模型
"""

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1, 
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

"""
Part 5. 绘图                
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
plt.savefig("02_Simple_GRU_Accuracy.png")
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("02_Simple_GRU_Loss.png")
plt.show()
