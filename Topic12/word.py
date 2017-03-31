# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:11:35 2017

@author: Qin
"""

f = open('filtered_words.txt', encoding = 'utf-8')
words = f.readlines()
f.close()

temp = input('请输入：')
for each_word in words:
    if each_word[:-1] in temp:
        print(temp.replace(each_word[:-1],'*' * len(each_word[:-1])))
        break
else:
    print(temp)