# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 19:05:05 2017

@author: Qin
"""

import re
import os

def find_word(path):
    files = os.listdir(path)
    word_re = re.compile(r'[\w\-\_\']+')
    word_dict = dict()
    
    for file in files:
        if os.path.isfile(file) and (os.path.splitext(file)[1] == '.txt'):
            with open(file,encoding='utf-8') as f:
                data = f.read()
                words = word_re.findall(data)
                for word in words:
                    try:
                        word_dict[word] += 1
                    except KeyError:
                        word_dict[word] = 1
                        
    word_list = sorted(word_dict.items(),key=lambda t : t[1],reverse=True)
    print(word_list)
    
if __name__ == '__main__':
    find_word(os.curdir)