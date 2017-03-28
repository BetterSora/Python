# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:46:42 2017

@author: Qin
"""

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    is_text = False
    def handle_starttag(self, tag, attrs):
        if tag =='div':
            self.is_text = True
        elif tag == 'br':
            print()
        else:
            self.is_text = False
    
    def handle_endtag(self, tag):
        pass
    
    def handle_data(self, data):
        temp = data.strip()
        if self.is_text and temp != '':
            print(data.strip())

if __name__ == '__main__':
    parser = MyHTMLParser()
    
    with open('3.html', encoding='utf-8') as f:
        data = f.read()
    
    parser.feed(data)
    