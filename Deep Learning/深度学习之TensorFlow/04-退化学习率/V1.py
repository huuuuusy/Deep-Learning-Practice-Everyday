"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
框架： TensorFlow-GPU == 1.13.1
介绍： 使用退化学习率，在训练速度和精度之间找到平衡
"""

"""
退化学习率
    如果learning rate比较大，则训练速度提升，但是精确度不够
    如果learning rate比较小，则训练精确度提升，但是会耗费更多时间
    退化学习率又叫学习率衰减，在训练初期使用大学习率增加训练速度，在训练进行到一定程度后降低学习率，提升精确度
"""
import tensorflow as tf

global_step = tf.Variable(0, trainable=False) # 标记循环次数
initial_learning_rate = 0.1 # 初始学习率
decay_steps = 10 # 每循环10次进行一次衰减
decay_rate = 0.9 # 每次衰减时，衰减系数是0.9
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=False) # staircase默认False,为True则无衰减功能

opt = tf.train.GradientDescentOptimizer(learning_rate) # 使用梯度下降算法

add_global = global_step.assign_add(1) # 定义一个op,将global_step加一完成计数　

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(20):
        g, rate = sess.run([add_global, learning_rate])
        print(g, rate)
"""
结果：
    0.1 --> 初始学习率
    1 0.09895193
    2 0.09791484
    3 0.09688862
    4 0.095873155
    5 0.094868325
    6 0.09387404
    7 0.092890166
    8 0.09191661
    9 0.09095325
    10 0.089999996 --> 0.1*0.9 = 0.09，每10次退化到上一次的90%
    11 0.08905673
    12 0.088123344
    13 0.08719975
    14 0.08628584
    15 0.0853815
    16 0.084486626
    17 0.08360115
    18 0.08272495
    19 0.08185792
    20 0.08099999 --> 0.09*0.9 = 0.081,每10次退化到上一次的90%
"""