#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:11:05 2018

@author: ly
"""

from keras import backend as K
import json

from utils import crnn_model
from utils import load_data



data_dir='../data/synth90k'

crnn=crnn_model.CRNNCTCNetwork('train',256,20,37,(32,100,3))
model=crnn.build_network()



char_map_dict = json.load(open('../data/char_map.json', 'r'))


txtdata=load_data.TextData(data_dir,char_map_dict,(100,32),9)

#model.compile(optimizer='adam',
#              loss='ctc')

if __name__=='__main__':
    pass