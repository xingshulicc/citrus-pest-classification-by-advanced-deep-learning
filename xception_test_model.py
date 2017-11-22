# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Sep 25 09:51:13 2017

@author: xingshuli
"""

'''
The default_size of input image is 299 and min_size = 71
Should note that the input preprocessing function is also different from
VGG16 and ResNet but same as Xception: the pixel value between -1 and 1
'''

import os
import numpy as np
import keras

from keras.layers import Input
from keras.layers import Dense
#from keras.layers import Dropout

from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping  

img_width, img_height = 200, 200  
input_tensor = Input(shape=(img_width, img_height, 3))

train_data_dir = os.path.join(os.getcwd(), 'data/train')
validation_data_dir = os.path.join(os.getcwd(), 'data/validation')

nb_train_samples = 5000
nb_validation_samples = 1000

num_class = 10
epochs = 100
batch_size = 20

#load pre_trained model

base_model = Xception(include_top = False, weights = 'imagenet', 
                      input_tensor = input_tensor, pooling = 'avg')

#visualize layer name and layer indices 
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

#reconstruct fully-connected layers
x = base_model.output
x = Dense(2048, activation='relu')(x)
#x = Dropout(0.25)(x)
pre_out = Dense(num_class, activation='softmax')(x)

#reconstruct the model
train_model = Model(base_model.input, outputs= pre_out, name='train_model')

#we will freeze the first 3 layers and fine-tune the rest
for layer in base_model.layers[:4]:
    layer.trainable = False
for layer in base_model.layers[4:]:
    layer.trainable = True

#fine-tune:the learning rate should be smaller 
sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
train_model.compile(loss='categorical_crossentropy', 
                   optimizer=sgd, metrics=['accuracy'])

train_model.summary()
#imgae data augmentation
train_datagen = ImageDataGenerator(rotation_range=30, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rescale=1. / 255,
                                   fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
    

#early-stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

#train the model on the new data
hist = train_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples //batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples //batch_size, 
    callbacks=[early_stopping])

#print acc and stored into acc.txt
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print val_acc and stored into val_acc.txt
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()

#evaluate the model
evaluation = train_model.evaluate_generator(validation_generator,
                                      steps=nb_validation_samples //batch_size)
                                      
print('Model Accuracy = %.4f' % (evaluation[1]))



#predict a category of input image
img_path = '/home/xingshuli/Desktop/test_pictures/citrus_swallowtail.jpeg'
img = image.load_img(img_path, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /=255
print('Input image shape:', x.shape)
preds = train_model.predict(x)
print('citrus_swallowtail:%f, forktailed_bush_katydid:%f, ground_beetle:%f, \
green_stink_bug:%f, green_leafhopper:%f, syrhip_fly:%f, dragon_fly:%f, \
mantis:%f, fruit_moth:%f, citrus_longicorn_beetle:%f' \
 %(preds[0][0], preds[0][1], preds[0][2], preds[0][3], preds[0][4], 
   preds[0][5], preds[0][6], preds[0][7], preds[0][8], preds[0][9]))


#the pre-processing for imagenet in Xception

#def preprocess_input(x):
#    x /= 255.
#    x -= 0.5
#    x *= 2.
#    return x


#if __name__ == '__main__':
#    model = Xception(include_top=True, weights='imagenet')
#
#    img_path = 'elephant.jpg'
#    img = image.load_img(img_path, target_size=(299, 299))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
#    print('Input image shape:', x.shape)
#
#    preds = model.predict(x)
#    print(np.argmax(preds))
#    print('Predicted:', decode_predictions(preds, 1))













