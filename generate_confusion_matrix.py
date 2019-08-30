# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Dec  5 15:38:41 2018

@author: xingshuli
"""
import os
import numpy as np
import shutil

#model_name = mobilenet_v2

file_name = 'misclassify.txt'
file_path = os.path.join(os.getcwd(), file_name)
line_list = []

with open(file_path, 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.split():
            line = eval(line)
            line_list.append(line)


#print(line_list)

label_key = 'target'
predict_key = 'real'
file_key = 'file'

num_classes = 24
width, height = 224, 224
wrong_images_folder = os.path.join(os.getcwd(), 'wrong_images_1')
if not os.path.isdir(wrong_images_folder):
    os.makedirs(wrong_images_folder)

data_path = os.path.join(os.getcwd(), 'image_Data/validation')
folders_name = os.listdir(data_path)
folders_name = np.array(folders_name)
folders_name = folders_name[np.argsort(folders_name)]

folders_path = []
for folder in folders_name:
    folder_path = os.path.join(data_path, folder)
    folders_path.append(folder_path)


folders_images = []
for path in folders_path:
    folder_images = os.listdir(path)
    num_images = len(folder_images)
    folders_images.append(num_images)
folders_images = np.array(folders_images, dtype = int)
folders_images = np.reshape(folders_images, (num_classes, 1))

diag_matrix = np.eye(num_classes, dtype = int)
diag_matrix = diag_matrix * folders_images

for d in line_list:
    actual_label = d[label_key][0]
    predict_label = d[predict_key][0]
    diag_matrix[actual_label, actual_label] = diag_matrix[actual_label, actual_label] - 1
    diag_matrix[actual_label, predict_label] = diag_matrix[actual_label, predict_label] + 1
    image_path = d[file_key]
    image_path = os.path.join(os.getcwd(), image_path.strip('./'))
    image_new_name = str(actual_label) + '_' + str(predict_label) + '_' + os.path.basename(image_path)
    image_new_path = os.path.join(wrong_images_folder, image_new_name)
    shutil.copy(image_path, image_new_path)
    
       
confusion_matrix = diag_matrix
##print(confusion_matrix)
np.save('weaklydensenet_confusion_matrix', confusion_matrix)