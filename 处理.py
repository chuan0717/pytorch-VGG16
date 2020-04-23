import imageio  #imageio用来读取图像
import torch
from skimage.transform import resize  #resize更改图像尺寸大小
import os
import skimage.io

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

mean = torch.FloatTensor(256*[256*[mean]])
std = torch.FloatTensor(256*[256*[std]])

def imgchange(path):
    image_path = path
    img = imageio.imread(image_path)
    img = resize(img, (256, 256))

    img = torch.FloatTensor(img)

    # img-mean：去均值 ， /std 除方差
    img = (img - mean) / std
    skimage.io.imsave(image_path, img)
    return img

filename = 'C:\\Users\\11138\\Desktop\\植物图片爬取'
list1 = os.listdir(filename)
for name in list1:
    newname = filename + '\\' + name
    list2 = os.listdir(newname)
    for i in list2:
        newnewname = newname + '\\' + i
        imgchange(newnewname)
