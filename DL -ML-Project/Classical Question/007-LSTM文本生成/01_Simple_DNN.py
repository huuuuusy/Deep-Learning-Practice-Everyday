"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

使用LSTM生成尼采的作品
"""

"""
Part 1. 数据准备
"""

import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

