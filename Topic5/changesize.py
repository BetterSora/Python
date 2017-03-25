# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:40:59 2017

@author: Qin
"""

import os
from PIL import Image

def changesize(im,name,weigth,heigth):
    '改变图片分辨率为Iphone5的分辨率'   
    im.thumbnail((weigth,heigth))
    savapic(im,name)

def savapic(im_re,name):
    '保存修改后图片'
    os.chdir('修改后图片')
    im_re.save(name+'.jpg')
    os.chdir(os.pardir)

if __name__ == '__main__':
    files = [file for file in os.listdir(os.curdir) if os.path.isfile(file) and (os.path.splitext(file)[1] == '.jpg')]
    
    try:
        os.mkdir('修改后图片')
    except Exception:
        print('文件夹已存在')
    
    for file in files:
        name = os.path.splitext(file)[0]
        with Image.open(file) as im:
            im_re = changesize(im,name,640,1136)
