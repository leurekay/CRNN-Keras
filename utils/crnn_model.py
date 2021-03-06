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
from keras.layers.core import Lambda

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :] 测试感觉没影响
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)




class CRNNCTCNetwork(object):
    def __init__(self, phase, hidden_num, layers_num, num_classes, input_tensor_shape):
        self.__phase = phase.lower()
        self.__hidden_num = hidden_num
        self.__layers_num = layers_num
        self.__num_classes = num_classes
        self.input_tensor=Input(name='the_input',shape=input_tensor_shape)
        
        return

    def __feature_sequence_extraction(self):
        
        
        input_tensor=self.input_tensor
        #first 2 conv layers
        x = Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu', kernel_initializer='he_normal',name='conv1')(input_tensor)
        x = MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool1')(x)

        x = Conv2D(128, (3, 3), strides=(1,1), padding='same', activation='relu', kernel_initializer='he_normal',name='conv2')(x)
        x = MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool2')(x)
        
        x = Conv2D(256, (3, 3), strides=(1,1), padding='same', activation='relu',kernel_initializer='he_normal', name='conv3')(x)
        x = Conv2D(256, (3, 3), strides=(1,1), padding='same', activation='relu', kernel_initializer='he_normal',name='conv4')(x)
        x = MaxPooling2D(pool_size=(2,1),strides=(2,1), name='pool3')(x)
        
        x = Conv2D(512, (3, 3), strides=(1,1), padding='same', activation='relu', kernel_initializer='he_normal',name='conv5')(x)
        x = BatchNormalization( name='bn1')(x)
        
        x = Conv2D(512, (3, 3), strides=(1,1), padding='same', activation='relu', kernel_initializer='he_normal',name='conv6')(x)
        x = BatchNormalization( name='bn2')(x)
        
        x = MaxPooling2D(pool_size=(2,1),strides=(2,1), name='pool4')(x)
        
        x = Conv2D(512, (2, 1), strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal',name='conv7')(x)
#        x=keras.backend.squeeze(x, axis=0)
#        x=keras.backend.expand_dims(x, axis=-1)
        x=Reshape((-1,512))(x)
        return x
    
    def __feature_sequence_extraction2(self):
        is_training = True if self.__phase == 'train' else False
        
        
        input_tensor=self.input_tensor
        #first 2 conv layers
        x = Conv2D(16, (3, 3), strides=(1,1), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

        x = Conv2D(16, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        
#        x = Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,1),strides=(2,1))(x)
        
        x = Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = MaxPooling2D(pool_size=(2,1),strides=(2,1))(x)
        
        x = Conv2D(32, (2, 1), strides=(1,1), padding='valid', activation='relu')(x)
#        x=keras.backend.squeeze(x, axis=0)
#        x=keras.backend.expand_dims(x, axis=-1)
        x=Reshape((-1,32), name='reshape1')(x)
        return x



    def build_network(self,max_label_length=None):
        if self.__phase in ['train','valid']:
            assert(max_label_length)
        
        x=self.__feature_sequence_extraction()
        
        x=Bidirectional(LSTM(units=self.__hidden_num,return_sequences=True,kernel_initializer='he_normal'), merge_mode='concat',name='bilstm1')(x)
#        x=LSTM(units=self.__hidden_num,return_sequences=True)(x)
        
        _,n_t,n_logit=x.shape.as_list()
#        x=keras.backend.expand_dims(x, axis=-1)
#        x=Conv2D(self.__num_classes, (1, n_logit), strides=(1,1), padding='valid', activation='softmax')(x)
        
        x=Dense(units=self.__num_classes,activation='softmax', kernel_initializer='he_normal',name='dense1')(x)
        
        
        
        labels = Input(name='the_labels', shape=[max_label_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])
        model = Model(inputs=[self.input_tensor, labels, input_length, label_length], outputs=[loss_out])
        if self.__phase != 'train':
            model=Model(inputs=self.input_tensor,outputs=x)
        return model
    
#    def decode(self):
##        input_length = Input(name='input_length', shape=[1], dtype='int64')
#        model=self.build_network()
#        input_image,input_length=model.input
#
#        softmax=model.output
#        decode=K.ctc_decode(softmax, input_length, greedy=False, beam_width=100, top_paths=1)
#        decode_model=Model(inputs=[input_image,input_length],outputs=decode)
#        return decode_model
        
        

def wrap_ctc_loss(y_true,y_pred):
        labels = y_true['the_labels']
        input_length = y_true['input_length']
        label_length = y_true['label_length']
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


if __name__=='__main__':
    crnn=CRNNCTCNetwork('test',256,20,37,(32,100,3))
    model=crnn.build_network(23)
#    decode_model=crnn.decode()
    print (model.summary())
#    plot_model(model, to_file='../data/model.jpg',show_shapes=True)