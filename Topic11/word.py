# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:40:20 2017

@author: Qin
"""

f = open('filtered_words.txt', encoding = 'utf-8')
words = f.readlines()
f.close()

temp = input('请输入：')
for each_word in words:
    if each_word in temp:
        print('Freedom')
        break
else:
    print('Human Rights')