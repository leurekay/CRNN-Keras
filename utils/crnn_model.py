#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:30:20 2018

@author: dirac
"""

import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K


from keras.models import Model
from keras.layers import Dense, Dropout, Flatten,Input,Activation,Reshape,Lambda
from keras.layers import Conv2D, MaxPooling2D,MaxPooling3D,Conv3D,Deconv2D,Deconv3D
from keras.layers import LSTM, Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

from keras.utils import np_utils  
from keras.utils import plot_model  


class CRNNCTCNetwork(object):
    def __init__(self, phase, hidden_num, layers_num, num_classes, input_tensor_shape=(32,128,3)):
        self.__phase = phase.lower()
        self.__hidden_num = hidden_num
        self.__layers_num = layers_num
        self.__num_classes = num_classes
        self.input_tensor=Input(shape=input_tensor_shape)
        
        return

    def __feature_sequence_extraction(self):
        is_training = True if self.__phase == 'train' else False
        
        
        input_tensor=self.input_tensor
        #first 2 conv layers
        x = Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

        x = Conv2D(128, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        
        x = Conv2D(256, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(1,2),strides=(2,2))(x)
        
        x = Conv2D(512, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(512, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = MaxPooling2D(pool_size=(1,2),strides=(2,2))(x)
        
        x = Conv2D(512, (2, 2), strides=(1,1), padding='valid', activation='relu')(x)
#        x=keras.backend.squeeze(x, axis=0)
#        x=keras.backend.expand_dims(x, axis=-1)
        x=Reshape((-1,512))(x)
        return x
    


    def build_network(self):
        x=self.__feature_sequence_extraction()
        
#        x=Bidirectional(LSTM(units=self.__hidden_num,return_sequences=True))(x)
        x=LSTM(units=self.__hidden_num,return_sequences=True)(x)
        
        model=Model(inputs=self.input_tensor,outputs=x)
        return model


if __name__=='__main__':
    crnn=CRNNCTCNetwork('train',256,20,26,(32,None,3))
    model=crnn.build_network()
    print (model.summary())
    plot_model(model, to_file='../data/model.jpg',show_shapes=True)