"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

对示例图像进行常见处理
"""
from imtools import *
from PIL import Image
from pylab import *

# 读取第一张图片并灰度化
im_path = 'Python-CV-Project/image_data'
im_list = get_imlist(im_path)
print(im_list[0])
im_test = array(Image.open(im_list[0]).convert('L'))
imshow(im_test)

# 显示灰度图像
figure()
gray()
contour(im_test, origin='image')

# 显示直方图信息
figure()
hist(im_test.flatten(),128)
show()

