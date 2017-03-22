# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:14:32 2017

@author: Qin
"""

import random
data = ['0','1','2','3','4',
        '5','6','7','8','9',
        'a','b','c','d','e',
        'f','g','h','i','j',
        'k','l','m','n','o',
        'p','q','r','s','t',
        'u','v','w','x','y',
        'z']
def get_coupon():
    length = len(data)
    coupons = ''
    for count in range(200):
        coupon = ''
        for i in range(10):
            temp = random.randint(0,length-1)
            coupon += data[temp]
        coupons += 'no.%d:%s\n' %(count+1,coupon)
        
    return coupons

if __name__ == '__main__':
    coupons = get_coupon()
    with open('coupon.txt','w') as f:
        f.write(coupons)
        
'''
import string
import random
def coupon_creator(digit):
    coupon=''
    for word in range(digit):
        coupon+=random.choice(string.ascii_uppercase + string.digits)
    return coupon
    
def two_hundred_coupons():
    data=''
    count=1
    for count in range(200):
        digit=12
        count+=1
        data+='coupon no.'+str(count)+'  '+coupon_creator(digit)+'\n'

    return data


coupondata=open('coupondata.txt','w')
coupondata.write(two_hundred_coupons())
coupondata.close()
'''