#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:43:49 2018

@author: dirac
"""

import os
import cv2

data_dir='../data/synth90k'
list_path=os.path.join(data_dir,'image_list.txt')

with open(list_path) as f:
    txt=f.readlines()
    txt=list(map(lambda x : x[:-1],txt))
    
image_path=os.path.join(data_dir,txt[-1])
#image_path='../data/synth90k/701/4/242_Snapdragon_72126.jpg'
image = cv2.imread(image_path)