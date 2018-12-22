#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:15:59 2018

@author: ly
"""

import json

CHAR_MAP_PATH='../data/char_map.json'
CHAR_MAP_DICT = json.load(open(CHAR_MAP_PATH, 'r'))
NUM_CLASS=len(CHAR_MAP_DICT)+1

TRAIN_DIR='../data/synth90k/train'
VALID_DIR='../data/synth90k/valid'
CHECKPOINT_DIR='../data/checkpoints'


#================model parameters====================
MAX_LABEL_LENGTH=12
INPUT_LENGTH=25
TRAIN_W=100
TRAIN_H=32



#====================================================





#===============train=================================
EPOCHS=30
BATCH_SIZE=32
