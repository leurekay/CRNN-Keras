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

w=135
h=32
input_length=w/4


check_dir='../data/checkpoints'
check_path=os.path.join(check_dir,'epoch:001-loss:24.174.h5')

data_dir='../data/synth90k/valid-tmp/611/2'
file_list=os.listdir(data_dir)
image_path=os.path.join(data_dir,file_list[2])
image = cv2.imread(image_path)
image = cv2.resize(image, (w, h))    
image=np.expand_dims(image,axis=0)
image=image/255.
input_length=np.array([[input_length]])

crnn=crnn_model.CRNNCTCNetwork('test',256,20,37,(h,w,3))
model=crnn.build_network()

model.load_weights(check_path, by_name=True)
pred=model.predict([image,input_length])

char_map_dict = json.load(open('../data/char_map.json', 'r'))
