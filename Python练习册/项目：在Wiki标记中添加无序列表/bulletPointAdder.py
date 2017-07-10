#! python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:32:40 2017

@author: Qin
"""

import pyperclip

test = pyperclip.paste()

lines = test.split('\n')
for i in range(len(lines)):
    lines[i] = '*' + lines[i]
text = '\n'.join(lines)

pyperclip.copy(text)