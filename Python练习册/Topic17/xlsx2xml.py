# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:47:43 2017

@author: Qin
"""

import openpyxl
from lxml import etree

def read_xlsx(filename):
    workbook = openpyxl.load_workbook('student.xlsx')
    worksheet = workbook.worksheets[0]
    datadict = {}
    for i in range(1, 4):
        temp = [worksheet.cell(row = i, column = j).value for j in range(2, 6)]
        datadict[str(i)] = temp
    
    return datadict
    
def save_xml(data):
    root = etree.Element('root')
    student = etree.SubElement(root, 'student')
    
    #添加注释和正文
    student.append(etree.Comment('学生信息表\n"id": [名字，数学，语文，英语]'))
    student.text = str(data)
    
    tree = etree.ElementTree(root)
    tree.write('student.xml', encoding = 'utf-8', pretty_print = True, xml_declaration = True)

if __name__ == '__main__':
    data = read_xlsx('student.xlsx')
    save_xml(data)