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
        'password':'',
        'db':'coupon',
        'charset':'utf8'
        }

#创建连接
conn = pymysql.connect(**config)

try:
    with conn.cursor() as cursor:
        with open('coupon.txt') as f:
            for line in f:
                 sql = 'INSERT INTO coupons (coupon_data) VALUES (%s)'
                 cursor.execute(sql, (line[:-1]))    
    conn.commit()

 
finally:
    conn.close()
    
'''
CREATE TABLE `coupons` (
    `coupon_data` varchar(255) COLLATE utf8_bin NOT NULL
);
'''