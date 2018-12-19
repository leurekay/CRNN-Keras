#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:11:05 2018

@author: ly
"""

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import json
import os

from utils import crnn_model
from utils import load_data

check_dir='../data/checkpoints'

data_dir='../data/synth90k'

crnn=crnn_model.CRNNCTCNetwork('train',256,20,37,(32,100,3))
model=crnn.build_network()



char_map_dict = json.load(open('../data/char_map.json', 'r'))


txtdata=load_data.TextData(data_dir,char_map_dict,(100,32),9,25)

model.compile(optimizer='adam',
              loss={'ctc': lambda y_true, y_pred: y_pred})

checkpoint=ModelCheckpoint(filepath=os.path.join(check_dir,'epoch:{epoch:03d}-loss:{loss:.3f}.h5'), 
                                monitor='loss', 
                                verbose=0, 
                                save_best_only=False, 
                                save_weights_only=True, 
                                period=1)



model.fit_generator(generator=txtdata.next_batch(64),
                    callbacks=[checkpoint],
                    steps_per_epoch=30,
                    epochs=3,)



if __name__=='__main__':
    pass