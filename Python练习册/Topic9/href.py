# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:07:14 2017

@author: Qin
"""

import re
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for (t, v) in attrs:
                if t == 'href':
                    if re.match(r'http(.)+', v):
                        print(v)
                        
if __name__ == '__main__':
    parser = MyHTMLParser()
    
    with open('test.html', encoding='utf-8') as f:
        html = f.read()
        
    parser.feed(html)