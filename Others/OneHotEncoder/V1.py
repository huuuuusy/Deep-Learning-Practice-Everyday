"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
介绍： sklearn中OneHotEncoder用法解析
参考： https://www.cnblogs.com/zhoukui/p/9159909.html
"""

from sklearn.preprocessing import OneHotEncoder

"""
class OneHotEncoder(n_values=None, categorical_features=None, categories=None, 
                    drop=None, sparse=True, dtype=np.float64, handle_unknown='error')
"""

enc = OneHotEncoder()

enc.fit([[0, 0, 3],
         [1, 1, 0],
         [0, 2, 1],
         [1, 0, 2]])

ans = enc.transform([[0, 1, 3]]).toarray()  # 如果不加toarray()，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果
print(ans) # [[1. 0. 0. 1. 0. 0. 0. 0. 1.]]
"""
第一列特征[0, 1, 0, 1]
    一共出现2种元素0/1:对于0，使用(1,0)编码;对于1,使用(0,1)编码
第二列特征[0, 1, 2, 0]
    一共出现3种元素0/1/2：对于0,使用(1,0,0)编码；对于1，使用(0,1,0)编码;对于2,使用(0,0,1)编码
第三列特征[3, 0, 1, 2]
    一共出现4中特征0/1/2/3:对于0, 使用(1,0,0,0)编码；对于1,使用(0,1,0,0)编码;对于2，使用(0,0,1,0)编码;对于3,使用(0,0,0,1)编码
最后输入[0,1,3],按照上面的编码方式，得到(1,0)(0,1,0)(0,0,0,1)
"""

"""
参数n_values
    表示每个特征使用几维的数值,可以由数据集自动推断，即几种类别就使用几位来表示；也可以人为指定
"""
enc = OneHotEncoder(n_values = [2,3,4]) # 人为指定每一列特征有几类数据
enc.fit([[0, 0, 3],
         [1, 1, 0]])

ans = enc.transform([[0, 2, 3]]).toarray() # 即使出现数据集中没有的数字(如第二列的2)，也可以参与编码
print(ans) # [[ 1.  0.  0.  0.  1.  0.  0.  0.  1.]]
