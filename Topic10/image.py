# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:47:31 2017

@author: Qin
"""

import string
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

#字母
letter = string.ascii_uppercase
#像素点颜色
pointcolor = lambda : (random.randint(0,255), random.randint(0,255), random.randint(0,255))
#字母颜色
fontcolor = lambda : (random.randint(0,255), random.randint(0,255), random.randint(0,255))
#字体
myfont = ImageFont.truetype(r'C:\windows\fonts\Arial.ttf', size=40)

im = Image.new('RGB', (300, 100), '#ffffff')

draw = ImageDraw.Draw(im)

for x in range(300):
    for y in range(100):
        draw.point((x, y), fill = pointcolor())

for i in range(4):
    draw.text((50 + 55 * i,30), random.choice(letter), font = myfont, fill = fontcolor())
    
im = im.filter(ImageFilter.BLUR)

im.save('image.jpg', 'jpeg')
