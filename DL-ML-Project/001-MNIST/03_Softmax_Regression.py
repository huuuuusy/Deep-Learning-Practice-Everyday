"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.36
框架： TensorFlow-GPU == 1.13.1
介绍： 本例使用最简单的机器学习模型--单一softmax regression对MNIST进行分类
      主要目的是熟悉TensorFlow的执行流程
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

"""
下载数据
"""
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print('MNIST train:\n' ,mnist.train.images)
print('Train shape:\n', mnist.train.images.shape)
print('Validation shape:\n', mnist.validation.images.shape)
print('Test shape:\n', mnist.test.images.shape)

img = mnist.train.images[1]
img = img.reshape(-1, 28)
pylab.imshow(img)
pylab.show

"""
构建模型
"""
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.matmul(x,W) + b)
# 损失函数，其中pred和y先进行一次交叉熵的运算，然后取均值
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

"""
训练模型
"""
training_epochs = 25
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path = "log/mnistmodel.ckpt"

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# Initializing OP

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print( " Finished!")
    
    
    # 测试模型
    # tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    #reduce_mean函数用于求tensor中的平均值， cast函数用于改变数据类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
    
    # 保存模型
    save_path = saver.save(sess, model_path)
    print("Model saved in files: %s" % save_path)

"""
读取模型
"""
print("starting 2nd session ...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)
    
    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accurracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict = {x:batch_xs})
    print(outputval, predv, batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()