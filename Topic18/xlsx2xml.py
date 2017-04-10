# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:30:29 2017

@author: Qin
"""

import openpyxl
from lxml import etree

def read_xlsx(filename):
    workbook = openpyxl.load_workbook(filename)
    worksheet = workbook.worksheets[0]
    datadict = {}
    for i in range(1, 4):
        datadict[str(i)] = worksheet.cell(row = i, column = 2).value
    
    return datadict

def save_xml(data):
    root = etree.Element('root')
    citys = etree.SubElement(root, 'citys')
    
    citys.append(etree.Comment('城市信息'))
    citys.text = str(data)
    
    tree = etree.ElementTree(root)
    tree.write('city.xml', encoding = 'utf-8', pretty_print = True, xml_declaration = True)

if __name__ == '__main__':
    data = read_xlsx('city.xlsx')
    save_xml(data)