# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Sat Aug 12 09:47:53 2017

@author: xingshuli
"""
#import numpy as np
import matplotlib.pyplot as plt
epochs = 201 #here, the epoch should add 1 

mat_list_acc = []
with open('/home/xingshuli/Desktop/acc.txt','r') as f:
    data = f.readlines()
    for line in data:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        mat_list_acc.append(num_float)
f.close()
y_axis = sum(mat_list_acc, [])

mat_list_val_acc = []
with open('/home/xingshuli/Desktop/val_acc.txt','r') as f1:
    data1 = f1.readlines()
    for line in data1:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        mat_list_val_acc.append(num_float)
f1.close()
y1_axis = sum(mat_list_val_acc, [])

x_axis = range(1,epochs)

plt.plot(x_axis, y_axis, 'r', label="acc")
plt.plot(x_axis, y1_axis, 'b', label="val_acc")
plt.legend(loc='best')
plt.show()


























