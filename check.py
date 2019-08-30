# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Sun Nov 26 14:43:35 2017

@author: xingshuli
"""
import os
import numpy as np

from keras.models import load_model

from keras.preprocessing import image

import matplotlib.pyplot as plt

#from keras.applications.mobilenet import relu6, DepthwiseConv2D
#from keras.utils.generic_utils import CustomObjectScope

val_path = './image_Data/validation/'

def dir_paths(path):
    dir_array = []
    for f in os.listdir(path):
        total_path = os.path.join(path, f)
        if os.path.isdir(total_path):
            dir_array.append(total_path)
    
    return sorted(dir_array)


def generate_array(path, shape = (224, 224), extension = '.jpeg'):
    image_arrays = None
    predict_list = []
    
    for f in os.listdir(path):
        if f.endswith(extension):
            total_path = os.path.join(path, f)
            predict_list.append(total_path)
            img = image.load_img(total_path)
            img = img.resize(shape)
     #load all image to numpy and convert to (None, 224, 224, 3)       
            img = image.img_to_array(img)
            img/=255
            img = img[np.newaxis, :]
            try:
                image_arrays = np.concatenate((image_arrays, img), 0)
            except:
                image_arrays = img
            
    print(image_arrays.shape)
    return image_arrays, predict_list

def predict(image_arrays, predict_list, target_index, category, draw_image = False):
    result = model.predict(image_arrays)
    wrong_predict = []
    for i in range(len(result)):
        real_index = np.argmax(result[i])
        if real_index != target_index:
            wrong_file = predict_list[i]
            wrong = {'file': wrong_file, 'target': (target_index, category[target_index].split('/')[-1]), 
                     'real': (real_index, category[real_index].split('/')[-1])}
            print(wrong)
            wrong_predict.append(wrong)
            img = image.array_to_img(image_arrays[i])
            if draw_image:
                plt.figure()
                plt.imshow(img)
                plt.axis('off')
                
    if draw_image:
        plt.show()
    
    return wrong_predict
    
#load model
#model_save_dir = os.path.join(os.getcwd(), 'Mobilenet_V2')
#model_name = 'keras_mobilenetv2_trained_model.h5'
#model_save_path = os.path.join(model_save_dir, model_name)
#with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
#    model = load_model(model_save_path)

model_save_dir = os.path.join(os.getcwd(), 'Weakly_DenseNet_v2')
model_name = 'keras_trained_model.h5'

model_save_path = os.path.join(model_save_dir, model_name)
model = load_model(model_save_path)

dir_array = dir_paths(val_path)
wrong_images = []
for i, path in enumerate(dir_array):
    image_arrays, predict_list = generate_array(path)
    wrong_image = predict(image_arrays, predict_list, i, dir_array)
    wrong_images.extend(wrong_image)
    
with open('./misclassify.txt', 'w') as f:
    for wrong in wrong_images:
        f.write(str(wrong) + '\n')


    
    
    
