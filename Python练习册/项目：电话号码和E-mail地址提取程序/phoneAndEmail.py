#! python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:03:57 2017

@author: Qin
"""

import pyperclip, re

phoneRegex = re.compile(r'''(
        (\d{3}|\(\d{3}\))?                # 区号
        (\s|-|\.)?                        # 分隔符
        (\d{3})                           # 前三个数字
        (\s|-|\.)?                        # 分隔符
        (\d{4})                           # 后四个数字
        (\s*(ext|x|ext.)\s*(\d{2,5}))?    # 分机号
        )''', re.VERBOSE)

emailRegex = re.compile(r'''(
        [a-zA-Z0-9._%+-]+                 # 用户名
        @                                 # @
        [a-zA-Z0-9.-]+                    # 域名
        (\.[a-zA-Z]{2,4})                 # dot-something
        )''', re.VERBOSE)

text = str(pyperclip.paste())
matches = []
for groups in phoneRegex.findall(text):
    phoneNum = '-'.join([groups[1], groups[3], groups[5]])
    if groups[8] != '':
        phoneNum += ' X' + groups[8]
    matches.append(phoneNum)
    
for groups in emailRegex.findall(text):
    matches.append(groups[0])
    
if len(matches) > 0:
    pyperclip.copy('\n'.join(matches))
    print('Copied to clipboard:')
    print('\n'.join(matches))
else:
    print('No phone numbers or email addresses found')