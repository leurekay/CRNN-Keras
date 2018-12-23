#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:11:05 2018

@author: ly
"""
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adadelta

import json
import os

from utils import crnn_model
from utils import load_data
from crnn import config

#tfconfig = tf.ConfigProto(allow_soft_placement=True)
#tfconfig.gpu_options.allow_growth = True
#set_session(tf.Session(config=tfconfig))


char_map_path=config.CHAR_MAP_PATH
char_map_dict=config.CHAR_MAP_DICT
num_class=config.NUM_CLASS

train_w=config.TRAIN_W
train_h=config.TRAIN_H
max_label_length=config.MAX_LABEL_LENGTH

checkpoint_dir=config.CHECKPOINT_DIR
train_dir=config.TRAIN_DIR
valid_dir=config.VALID_DIR

epochs= config.EPOCHS
batch_size=config.BATCH_SIZE
init_epoch=15  #identical with the prefix  of load_weights_path
load_weights_path=os.path.join(checkpoint_dir,'015-loss:0.454-val_loss:2.118.h5')
#load_weights_path=None

crnn=crnn_model.CRNNCTCNetwork('train',256,20,num_class,(train_h,None,3))
model=crnn.build_network(max_label_length=max_label_length)
if load_weights_path:
    model.load_weights(load_weights_path,by_name=True)


#char_map_dict = json.load(open('../data/char_map.json', 'r'))


txtdata=load_data.TextData(train_dir,char_map_dict,(train_w,train_h),max_label_length,25)
txtdata_valid=load_data.TextData(valid_dir,char_map_dict,(train_w,train_h),max_label_length,25)



model.compile(optimizer=Adadelta(),
              loss={'ctc': lambda y_true, y_pred: y_pred})

checkpoint=ModelCheckpoint(filepath=os.path.join(checkpoint_dir,'{epoch:03d}-loss:{loss:.3f}-val_loss:{val_loss:.3f}.h5'), 
                                monitor='loss', 
                                verbose=0, 
                                save_best_only=False, 
                                save_weights_only=True, 
                                period=1)



model.fit_generator(generator=txtdata.next_batch(batch_size),
                    callbacks=[checkpoint],
                    steps_per_epoch=txtdata.__nums__/batch_size,
                    validation_data=txtdata_valid.next_batch(batch_size),
                    validation_steps=txtdata_valid.__nums__/batch_size,
                    epochs=epochs,
                    initial_epoch=init_epoch)



if __name__=='__main__':
    pass