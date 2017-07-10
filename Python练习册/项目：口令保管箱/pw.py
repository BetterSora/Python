#! python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:05:20 2017

@author: Qin
"""

PASSWORDS = {'email': 'fdsjkfhskjdnfbnruiuhvxc',
             'blog': 'dasiuvoprnejbguidsf',
             'luggage': '12345'}

import sys, pyperclip

if len(sys.argv) < 2:
    print('Usage: python pw.py [account] - copy account password')
    sys.exit()
    
# 第一个参数是文件名
account = sys.argv[1]

if account in PASSWORDS:
    pyperclip.copy(PASSWORDS[account])
    print('Password for ' + account + ' copied to clipboard.')
else:
    print('There is no account named ' + account)