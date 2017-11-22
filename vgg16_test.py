# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Sep  4 09:25:28 2017

@author: xingshuli
"""
import keras
import numpy as np
import warnings
import os

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
#from keras.layers import Dropout
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

#from keras.callbacks import EarlyStopping 

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16(include_top=True, weights='imagenet', input_tensor=None, 
          input_shape=None, pooling=None, classes=1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    #determine proper input shape
    input_shape = _obtain_input_shape(input_shape, default_size=224, 
                                      min_size=48, 
                                      data_format=K.image_data_format(), 
                                      include_top=include_top)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    #Block_1
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)
    
    #Block_2
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)
    
    #Block_3
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(x)
    
    #Block_4
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)
    
    #Block_5
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(x)
    
    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        
    #Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.   
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    #create model
    model = Model(inputs, x, name='vgg16')
    
    #load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                              
    return model
        
        
#Recreate my own vgg16 model        
img_width, img_height = 200, 200  
input_tensor = Input(shape=(img_width, img_height, 3))

train_data_dir = os.path.join(os.getcwd(), 'data/train')
validation_data_dir = os.path.join(os.getcwd(), 'data/validation')

nb_train_samples = 5000
nb_validation_samples = 1000

num_class = 10
epochs = 100
batch_size = 20

base_model = VGG16(include_top=False, weights='imagenet', 
                   input_tensor=input_tensor, pooling='avg')

for i, layer in enumerate(base_model.layers):
    print(i, layer.name) 

x = base_model.output

#Rebuild the fully connected layers
#x = Flatten()(x)
#x = Dense(4096, activation='relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(4096, activation='relu')(x)
#x = Dropout(0.5)(x)
#pre_out = Dense(num_class, activation='softmax')(x)

x = Dense(512, activation='relu')(x)
pre_out = Dense(num_class, activation='softmax')(x)

#we will train the model again
train_model = Model(base_model.input, outputs= pre_out, name='train_model')

## we assume that all layers can be trainable
#for layer in base_model.layers:
#    layer.trainable = True

for layer in base_model.layers[:4]:
    layer.trainable = False
for layer in base_model.layers[4:]:
    layer.trainable = True

sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
train_model.compile(loss='categorical_crossentropy', 
                   optimizer=sgd, metrics=['accuracy'])
                   
                   
train_model.summary()
##preprocessing
train_datagen = ImageDataGenerator(samplewise_center=False,
                                   rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
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
#early_stopping = EarlyStopping(monitor='val_loss', patience=3)

hist = train_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples //batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples //batch_size
    )

#print(hist.history['acc'])
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print(hist.history['val_acc'])
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()

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

            
##if __name__ == '__main__':
##    model = VGG16(include_top=True, weights='imagenet')
##
##    img_path = '/home/xingshuli/Desktop/elephant.jpeg'
##    img = image.load_img(img_path, target_size=(224, 224))
##    x = image.img_to_array(img)
##    x = np.expand_dims(x, axis=0)
##    x = preprocess_input(x)
##    print('Input image shape:', x.shape)
##
##    preds = model.predict(x)
##    print('Predicted:', decode_predictions(preds))        
##        
        
    
    
            
            
        
        
    
    
    
            
    
















































































































