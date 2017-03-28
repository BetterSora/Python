# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:30:15 2017

@author: Qin
"""

import os

def getfiles():
    '获取文件名称'
    files = os.listdir(os.curdir)
    file_list = [file for file in files if os.path.isfile(file) and os.path.splitext(file)[1] == '.py']
 
    return file_list

def add2dict(value):
    '计数器'
    try:
        num_dict[value] += 1
    except KeyError:
        num_dict[value] = 1

def count_node(name):
    '统计代码数目'
    with open(name,encoding='utf-8') as f:
        for line in f:
            if line == '\n':             
                add2dict('linesep')
            elif line.startswith('#'):
                add2dict('note')
            else:
                add2dict('code')

if __name__ == '__main__':
    path = 'F:\pythoncodes'
    os.chdir(path)
    num_dict = dict()
    file_list = getfiles()
    
    for each_file in file_list:
        count_node(each_file)
        
    num_list = sorted(num_dict.items(),key=lambda x : x[1],reverse=True)
    print(num_list)