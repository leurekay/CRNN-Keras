#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 22:17:51 2018

@author: dirac
"""

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import json
import os
import cv2
import numpy as np

from utils import crnn_model
from utils import load_data
from crnn import config

h=config.TRAIN_H

check_dir=config.CHECKPOINT_DIR
check_path=os.path.join(check_dir,'epoch:002-loss:23.514-val_loss:23.822.h5')

data_dir='../data/synth90k/train/1851/1'
file_list=os.listdir(data_dir)
image_path=os.path.join(data_dir,file_list[14])


image_raw = cv2.imread(image_path)
height,width,_=image_raw.shape
w=int((h*width)/height)
input_length=int(w/4)

image = cv2.resize(image_raw, (w, h))    
image=np.expand_dims(image,axis=0)
image=image/255.
input_length=np.array([input_length])



crnn=crnn_model.CRNNCTCNetwork('test',256,20,37,(h,w,3))
model=crnn.build_network()
model.load_weights(check_path, by_name=True)


y_pred=model.predict(image)
y_pred_labels_tensor_list, prob = keras.backend.ctc_decode(y_pred, input_length, greedy=True) # 使用的是最简单的贪婪算法
y_pred_labels_tensor = y_pred_labels_tensor_list[0]
y_pred_labels = keras.backend.get_value(y_pred_labels_tensor) # 现在还是字符编码
y_pred_prob = keras.backend.get_value(prob)



char_map_dict = json.load(open('../data/char_map.json', 'r'))
