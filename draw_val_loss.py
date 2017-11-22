# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Sep 22 11:51:53 2017

@author: xingshuli
"""

import matplotlib.pyplot as plt
epochs = 101 #here, the epoch should add 1 

mat_val_loss = []
with open('/home/xingshuli/Desktop/val_loss.txt','r') as f:
    data = f.readlines()
    for line in data:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        mat_val_loss.append(num_float)
f.close()
y_axis = sum(mat_val_loss, [])


x_axis = range(1,epochs)

plt.plot(x_axis, y_axis, 'r', label="val_loss")
plt.legend(loc='best')
plt.show()
