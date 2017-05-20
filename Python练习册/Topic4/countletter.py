# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:09:31 2017

@author: Qin
"""

import re

def count(name):
    reg = re.compile(r'[a-zA-Z]+')
    letter_dict = dict()
    with open(name) as f:
        data = f.read()
        wordlist = reg.findall(data)
        for each in wordlist:
            for word in each:
                try:
                    letter_dict[word] += 1
                except KeyError:
                    letter_dict[word] = 1
                    
    return letter_dict

if __name__ == '__main__':
    letter_dict = count('letter.txt')
    list1 = [(k,v) for k,v in letter_dict.items()]
    list1.sort()
    for each in list1:
        print(each[0],each[1])