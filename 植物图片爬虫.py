from selenium import webdriver
import requests 
import time
import re
import os
import random
import csv
import pandas as pd

# 目标文件位置
file_path = 'C:\\Users\\11138\\Desktop\\毕设\\测试或运行要放桌面的\\输出\\'
file_list = os.listdir('C:\\Users\\11138\\Desktop\\毕设\\测试或运行要放桌面的\\输出')

# 随机获取20种植物名称
l = {}
def names():
    i = 0
    while i < 20:
        x = random.randrange(0 , 200 , 1)
        f = open(file_path + file_list[x] , encoding = 'utf-8')
        df = pd.read_csv(f , error_bad_lines=False)
        y = random.randrange(0 , df.index.size - 1 , 1)
        m = df.iloc[y , 1]
        n = df.iloc[y , 0]
        if n in ['', ' ','   '] or type(n) != str:
            pass
        else:    
            l[m] = n
            i += 1
    return l

names()

# 爬虫函数
def cathch_photo(url, plant_id, plant_name):
    # Referer必须填写，否则会被网站识别为机器人
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36', 'Referer': url}
    # 创建并定位植物名称文件夹
    os.mkdir('C:\\Users\\11138\\Desktop\\植物图片爬取\\' + plant_name)
    os.chdir('C:\\Users\\11138\\Desktop\\植物图片爬取\\' + plant_name)
    num = 0
    n = 0
    # 由于PPBC是拉动式网站，故观察图片地址网页构成设置循环读取
    while(num >= 0):
        print('正在爬取第' + str(num + 1) + "页")
        res = requests.get('http://ppbc.iplant.cn/ashx/getphotopage.ashx?page=' + str(num) + '1&n=2&group=sp&cid=' + plant_id)
        res.encoding='utf-8' 
        html=res.text
        # 匹配植物图片地址
        chapter_photo_list=re.findall("<img  alt='.*?' src='http://img1.iplant.cn/image2/.*?jpg' />", html)
        if chapter_photo_list != []:
            for chapter_photo in chapter_photo_list:
                x = chapter_photo.find('http')
                url2 = chapter_photo[x : -4]
                r = requests.get(url2, headers = headers)
                n += 1
                name = str(n)
                # 保存图片
                open('C:\\Users\\11138\\Desktop\\植物图片爬取\\' + plant_name + '\\' + name + '.jpg', 'wb').write(r.content)
        else:
            break
        num += 1

# 为实现搜索功能，在这里模拟Chrome浏览器搜索行为
opt = webdriver.ChromeOptions()
opt.set_headless()
driver = webdriver.Chrome(options=opt)
d = 0
for i in l.keys():
    d += 1
    print('开始爬取第' + str(d) + '种植物')
    driver.get('http://ppbc.iplant.cn')
    time.sleep(2)
    # 模拟点击和搜索操作
    driver.find_element_by_id('txt_key1').send_keys(i)
    driver.find_element_by_xpath("//input[@value='检索图片']").click()
    # 防止网页刷新太慢无法获取，设置等待时间
    time.sleep(2)
    url = driver.current_url
    plant_id = url[25:]
    plant_name = l[i]
    cathch_photo(url = url, plant_id = plant_id, plant_name = plant_name)
    print(i + ':' + '爬取完毕')