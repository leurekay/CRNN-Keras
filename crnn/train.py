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

import json
import os

from utils import crnn_model
from utils import load_data
from crnn import config

#tfconfig = tf.ConfigProto(allow_soft_placement=True)
#tfconfig.gpu_options.allow_growth = True
#set_session(tf.Session(config=tfconfig))


CHAR_MAP_PATH=config.CHAR_MAP_PATH
CHAR_MAP_DICT=config.CHAR_MAP_DICT
NUM_CLASS=config.NUM_CLASS

TRAIN_W=config.TRAIN_W
TRAIN_H=config.TRAIN_H
MAX_LABEL_LENGTH=config.MAX_LABEL_LENGTH

CHECKPOINT_DIR=config.CHECKPOINT_DIR
TRAIN_DIR=config.TRAIN_DIR
VALID_DIR=config.VALID_DIR

EPOCHS= config.EPOCHS
BATCH_SIZE=config.BATCH_SIZE


crnn=crnn_model.CRNNCTCNetwork('train',256,20,NUM_CLASS,(TRAIN_H,None,3))
model=crnn.build_network(max_label_length=MAX_LABEL_LENGTH)



#char_map_dict = json.load(open('../data/char_map.json', 'r'))


txtdata=load_data.TextData(TRAIN_DIR,CHAR_MAP_DICT,(TRAIN_W,TRAIN_H),MAX_LABEL_LENGTH,25)
txtdata_valid=load_data.TextData(VALID_DIR,CHAR_MAP_DICT,(TRAIN_W,TRAIN_H),MAX_LABEL_LENGTH,25)



model.compile(optimizer='adam',
              loss={'ctc': lambda y_true, y_pred: y_pred})

checkpoint=ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR,'epoch:{epoch:03d}-loss:{loss:.3f}.h5'), 
                                monitor='loss', 
                                verbose=0, 
                                save_best_only=False, 
                                save_weights_only=True, 
                                period=1)



model.fit_generator(generator=txtdata.next_batch(BATCH_SIZE),
                    callbacks=[checkpoint],
                    steps_per_epoch=txtdata.__nums__/BATCH_SIZE,
                    validation_data=txtdata_valid.next_batch(BATCH_SIZE),
                    validation_steps=txtdata_valid.__nums__/BATCH_SIZE,
                    epochs=EPOCHS)



if __name__=='__main__':
    pass