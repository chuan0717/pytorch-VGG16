# -*- coding: utf-8 -*-
import os,shutil
from sklearn.model_selection import train_test_split

def splitDir(dirPath,random_state):
    path_type=['测试', '训练']
    class_arr=[]
    class_name_arr=[]
    class_tmp=[]
    name_dir="name_dir"
    for (root, dirs, files) in os.walk(dirPath):
        if files:
            for f in files:
                if name_dir not in root:
                    #切换下一个分类时，将上一个分类的数据存入
                    if name_dir!="name_dir":#第一次运行即刻进入，故需判断
                        print('【%s】已被读取'%name_dir)
                        class_arr.append(class_tmp)
                        class_name_arr.append(name_dir)
                    name_dir=root.split('\\')[-1]
                    class_tmp=[]
                path = os.path.join(root,f)
                #删除根路径，消除不同根目录名称的影响
                path = path.replace(dirPath,'')
                class_tmp.append(path)
    #最后一个分类执行结束，没有被加入
    class_arr.append(class_tmp)
    class_name_arr.append(name_dir)
    print('【%s】已被读取'%name_dir)
    for class_tmp,class_name in zip(class_arr,class_name_arr):
        #将数据划分为训练集、测试集两部分    使用随机数种子，确保可以复现
        train, test = train_test_split(class_tmp, test_size = 0.2,random_state=random_state)
        data_split=[test, train]
        print('【%s】已被复制'%class_name)
        for data,dtype in zip(data_split,path_type):
            for path in data:
                path = dirPath + path
                fileName = path.replace(dirPath, mainPath + dtype + '\\')
                #分离文件名和路径
                fpath, fname = os.path.split(fileName)
                if not os.path.exists(fpath):
                    #创建路径
                    os.makedirs(fpath)
                shutil.copyfile(path,fileName)

mainPath = 'C:\\Users\\11138\\Desktop\\'
tainPath = 'C:\\Users\\11138\\Desktop\\训练'
testPath = 'C:\\Users\\11138\\Desktop\\测试'
os.makedirs(tainPath)
os.makedirs(testPath)
splitDir ('C:\\Users\\11138\\Desktop\\植物图片爬取\\', random_state=12345)