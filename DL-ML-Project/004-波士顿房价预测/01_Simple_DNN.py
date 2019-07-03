""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
框架： TensorFlow-GPU == 1.13.1
      keras == 2.2.3
介绍： 波士顿房价预测数据包含404个训练样本，102个测试样本
      预测房价是一个回归问题
"""
 
# 下载数据
from keras.datasets import boston_housing
(x_train, y_train), (x_test,  y_test) = boston_housing.load_data()

print('train data shape:', x_train.shape)
print('test data shape:', x_test.shape)
print('train label shape:', y_train.shape)
print('test label shape:', y_test.shape)
print('train label:', y_train)

# 数据标准化
# 使取值范围很大的数据可以转化到均值为0, 标准差为1的范围内
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

# 对测试数据的标准化使用的是训练数据的均值和方差
x_test -= mean
x_test /= std

# 构建网络
from keras import layers
from keras import models

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape = (x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) # 回归问题不需要激活函数
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# 使用k折验证的方法评估训练效果
import numpy as np 

k = 4 # k是数据将被分成的折数
num_val_samples = len(x_train)//k # 每一折的数据量
num_epochs = 80 
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    x_val = x_train[i*num_val_samples : (i+1)*num_val_samples]
    y_val = y_train[i*num_val_samples : (i+1)*num_val_samples]

    partial_x_train = np.concatenate(
    [x_train[: i*num_val_samples], x_train[(i+1)*num_val_samples :]], axis=0)
    partial_y_train = np.concatenate(
    [y_train[: i*num_val_samples], y_train[(i+1)*num_val_samples :]], axis=0)

    model = build_model()
    history = model.fit(partial_x_train, partial_y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(len(average_mae_history))

# 绘制每轮的验证集MAE
import matplotlib.pyplot as plt

# 平滑曲线
# 删除前10个数据点，因为取值范围与其他点不同
# 将每个数据点替换为前面数据点的指数移动平均值，以得到光滑曲线

def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
        
    return  smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig("01_Simple_DNN_MAE.png")
plt.show()

# 在测试集上验证
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
print('test mae score:', test_mae_score)