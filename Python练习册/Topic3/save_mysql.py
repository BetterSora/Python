# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:19:52 2017

@author: Qin
"""

import pymysql.cursors

config = {
        'host':'localhost',
        'port':3306,
        'user':'root',
        'charset':'utf8'
        }

#创建连接
conn = pymysql.connect(**config)

try:
    with conn.cursor() as cursor:
        #创建数据库
        try:
            cursor.execute('create database if not exists coupons')
        except Exception:
            print('create error')
        #选择数据库
        conn.select_db('coupons')
        #创建table
        cursor.execute('create table if not exists coupon_data(code char(255))')
        with open('coupon.txt') as f:
            for line in f:
                 sql = 'insert into coupon_data values (%s)'
                 cursor.execute(sql, (line[:-1]))    
    conn.commit()

 
finally:
    conn.close()
    
