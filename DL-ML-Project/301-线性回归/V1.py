"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 使用tensorflow进行线性回归
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
准备数据
"""
# 生成训练数据，train_Y是y=2x加上随机误差
train_X = np.linspace(-1, 1, 100)
train_Y = 2*train_X + np.random.randn(*train_X.shape)*0.3

plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.savefig("original data.png")
plt.show()

"""
创建模型
"""
X = tf.placeholder('float')
Y = tf.placeholder('float')
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
# 前向结构
z = tf.multiply(X,W) + b
tf.summary.histogram('z', z)
# 反向优化
cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
tf.summary.scalar('loss function', cost)

"""
训练模型
"""
# 初始化所有变量
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20
display_step = 2
# 保存模型,max_to_keep=1表示最多只保存一个检查点文件
saver = tf.train.Saver(max_to_keep=1)
savedir = 'log/'

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# 启动session
with tf.Session() as sess:
    sess.run(init)
    # 合并所有summary
    merged_summary_op = tf.summary.merge_all()
    # 创建summary_writer用于写文件
    summary_writer = tf.summary.FileWriter('log/mnist_with_summary', sess.graph)

    plotdata = {'batchsize':[], 'loss':[]}
    for epoch in range(training_epochs):
        for(x,y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})
            summary_str = sess.run(merged_summary_op,feed_dict={X:x, Y:y})
            summary_writer.add_summary(summary_str, epoch)
        
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print('Epoch:', epoch+1, 'cost = ', loss, 'W = ', sess.run(W), 'b = ', sess.run(b))
            if not (loss == 'NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
            saver.save(sess, savedir + 'linearmodel.cpkt',global_step=epoch)
    print('Finished!')
    print('cost = ', sess.run(cost, feed_dict={X:train_X, Y:train_Y}), 'W = ', sess.run(W), 'b = ', sess.run(b))


    # 图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.savefig("original data with fitted line.png")
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.savefig("Minibatch run vs. Training loss.png")
    plt.show()

    # 测试
    print ("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))

"""
载入模型
"""
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(savedir + 'linearmodel.cpkt', None, True)

load_epoch = 18
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    kpt = tf.train.latest_checkpoint(savedir)
    if kpt != None:
        saver.restore(sess2, kpt)
    # 测试
    print('x=0.2, z=', sess2.run(z, feed_dict={X: 0.2}))
    
