# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:57:45 2017

@author: Qin
"""

import urllib.request
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            if ('class', 'BDE_Image') in attrs:
                for (u, v) in attrs:
                    if u == 'src':
                        imglist.append(v)

def gethtml(url):
    head = {}  
    head['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:52.0) Gecko/20100101 Firefox/52.0' 
    req = urllib.request.Request(url, headers = head)
    response = urllib.request.urlopen(req)
    
    return response

def saveimg(url, name):
    response = gethtml(url)
    with open(name, 'wb') as f:
        f.write(response.read())

if __name__ == '__main__':
    imglist = []
    url = 'http://tieba.baidu.com/p/2166231880'
    response = gethtml(url)
    html = response.read().decode('utf-8')
    parser = MyHTMLParser()
    parser.feed(html)
    
    for url in imglist:
        name = url.split('/')[-1]
        saveimg(url, name)