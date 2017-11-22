# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Aug 23 09:46:37 2017

@author: xingshuli
"""
import numpy as np
import warnings

import keras
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
#from keras.layers import Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.callbacks import EarlyStopping 

import keras.backend as K

from keras.utils import layer_utils
from keras.utils.data_utils import get_file
#from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

import os

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters1, (1,1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters2, kernel_size, 
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
    
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters1, (1,1), strides=strides, 
               name=conv_name_base + '2a')(input_tensor)
    
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters2, kernel_size, padding='same', 
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    shortcut = Conv2D(filters3, (1,1), strides=strides, 
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
    
def ResNet50(include_top=True, weights='imagenet', input_tensor=None, 
             input_shape=None, pooling=None, classes=1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    
    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64, (7,7), strides=(2,2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1,1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = AveragePooling2D((7,7), name='avg_pool')(x)
    
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    #Ensure that the model takes into account any potential predecessors of 'input_tensor'
    
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    #create model
    model = Model(inputs, x, name='resnet50')
    
    #load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
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


img_width, img_height = 200, 200  
input_tensor = Input(shape=(img_width, img_height, 3))

train_data_dir = os.path.join(os.getcwd(), 'data/train')
validation_data_dir = os.path.join(os.getcwd(), 'data/validation')

nb_train_samples = 5000
nb_validation_samples = 1000

num_class = 10
epochs = 100
batch_size = 20
#batch_size_1 = 8
#batch_size_2 = 4

save_dir = os.path.join(os.getcwd(), 'resnet_model')
model_name = 'keras_resnet_trained_model.h5'

#visualize layer name and layer indices 

base_model = ResNet50(include_top=False, weights='imagenet', 
                      input_tensor=input_tensor, pooling='avg')
for i, layer in enumerate(base_model.layers):
    print(i, layer.name) 

x = base_model.output
#add fully-connected layers
x = Dense(2048, activation='relu')(x)
#x = Dropout(0.5)(x)
pre_out = Dense(num_class, activation='softmax')(x)
#the model we will train
train_model = Model(base_model.input, outputs= pre_out, name='train_model') 
#Determine how many layers we should freeze, i.e we will freeze the first
#5 layers and fine-tune the rest
for layer in base_model.layers[:6]:
    layer.trainable = False
for layer in base_model.layers[6:]:
    layer.trainable = True
#fine-tune:the learning rate should be smaller 
sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
train_model.compile(loss='categorical_crossentropy', 
                   optimizer=sgd, metrics=['accuracy'])

train_model.summary()
#imgae data generation
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

#train the model on the new data for a few epochs
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

evaluation = train_model.evaluate_generator(validation_generator,
                                      steps=nb_validation_samples //batch_size)

print('Model Accuracy = %.4f' % (evaluation[1]))                
                   

#save model 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
train_model.save(model_path)
print('save trained model at %s' %model_path)


#predict a category of input image
img_path = '/home/xingshuli/Desktop/test_pictures/green_stink_bug.jpeg'
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
   





#if __name__ == '__main__':
#    model = ResNet50(include_top=True, weights='imagenet')
#
#    img_path = '/home/xingshuli/Desktop/elephant.jpeg'
#    img = image.load_img(img_path, target_size=(224, 224))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
#    print('Input image shape:', x.shape)
#
#    preds = model.predict(x)
#print('Predicted:', decode_predictions(preds))            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    
    












































































