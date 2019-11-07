"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
框架： TensorFlow-GPU == 1.13.1
介绍： 使用softmax计算loss
"""

import tensorflow as tf

"""
交叉熵实验
    假设有一个标签labels和一个网络输出值logits,对两个值进行如下三次实验
        两次softmax实验：对logits分别进行1次和2次softmax
        交叉熵实验：将上一实验的两个值分别进行softmax_cross_entropy_with_logits
        自建公式实验:将2次softmax的值放到组合公式中得到正确的值
"""

labels = [[0, 0, 1], [0, 1, 0]] # 标签
logits = [[2, 0.5, 6], [0.1, 0, 3]] # 网络输出，第一类和标签相符，第二类和标签不符

# 对输出计算两次softmax
logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

# 计算交叉熵
result1 = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits_scaled)
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)

# 打印结果
with tf.Session() as sess:
    print('original logits = ', logits, '\n') # [[2, 0.5, 6], [0.1, 0, 3]]

    # 经过一次softmax后，数组中最大的数值会接近1,其余接近0,总和为1
    # [[0.01791432 0.00399722 0.97808844][0.04980332 0.04506391 0.90513283]]
    print('logits_scaled = ', sess.run(logits_scaled), '\n')

    # 经过两次softmax后，概率会发生变化
    # 因为第一次softmax的结果使得所有数在(0,1)之间分布，所以第二次之前输入的差值没有原始的大，所以本次之后的结果数值会变得更接近
    # [[0.21747023 0.21446465 0.56806517][0.2300214  0.22893383 0.5410447 ]]
    print('logits_scaled2 = ', sess.run(logits_scaled2), '\n') 

    # 计算交叉熵
    # 因为结果中第一类和标签相符，第二类和标签不符，所以第一类的交叉熵比较小，第二类的交叉熵比较大 
    # [0.02215516 3.0996735 ] 　　　
    print('result1 = ', sess.run(result1), '\n') 

    # result2是将logits进行一次softmax后再和labels计算交叉熵
    # 比较result1和result2，说明使用tf.nn.softmax_cross_entropy_with_logits时，logits无需进行softmax转换
    # 若传入softmax转换后的logits(logits_scaled),则相当于进行了两次softmax计算
    # [0.56551915 1.4743223 ] 
    print('result2 = ', sess.run(result2), '\n') 

    # 若传入tf.nn.softmax_cross_entropy_with_logits之前logits已经被softmax转换为logits_scaled
    # 此时需要自行构建loss函数而不是直接使用tf.nn.softmax_cross_entropy_with_logits
    # 否则相当于转换两次
    # [0.02215518 3.0996735 ] 
    print('result3 = ', sess.run(result3), '\n')

"""
one-hot编码
    比较非标准one-hot编码(总和为1,但是数组中的值不是1或者0)和标准one-hot编码在计算交叉熵时的差异
"""
labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]] # 总和为1，但是元素数值不是0或者1
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits= logits)
with tf.Session() as sess:
    print('result4 = ', sess.run(result4), '\n') # [2.1721554 2.7696736] 

# 标准one-hot编码的输出结果是[0.02215516 3.0996735] 
# 非标准的one-hot编码输出的结果是[2.1721554 2.7696736] 
# 非标准one-hot编码的labels和logits之间的交叉熵差异比标准one-hot编码的小
# 说明非标准one-hot编码无法较好的表达分类正确和分类错误的交叉熵之间的差异

"""
sparse交叉熵
    sparse_softmax_cross_entropy_with_logits针对非标准one-hot编码设计
    可以对非标准one-hot计算交叉熵，解决上面的问题
    sparse_softmax_cross_entropy_with_logits的样本真实值和预测值不需要one-hot编码
    但是要求分类的个数一定是从0开始
"""
labels = [2,1]
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print('result5 = ', sess.run(result5), '\n') 

# [0.02215516 3.0996735 ]和标准one-hot编码一致
# sparse交叉熵已经包含将非标准one-hot转化为标砖one-hot的操作

"""
计算loss
"""
loss = tf.reduce_mean(result1)
with tf.Session() as sess:
    print ("loss=",sess.run(loss)) # 1.5609143
    
labels = [[0,0,1],[0,1,0]]    
# -tf.reduce_sum(labels * tf.log(logits_scaled),1)等价于softmax_cross_entropy_with_logits
loss2 = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits_scaled),1) )
with tf.Session() as sess:
    print ("loss2=",sess.run(loss2)) # 1.5609143