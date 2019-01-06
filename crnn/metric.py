#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 19:03:49 2019

@author: dirac
"""
import os

import inference
import config
from utils import load_data


def editDistance(s1, s2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    s1=' '+s1
    s2=' '+s2
    
    m=len(s1)
    n=len(s2)
    
    tab=[[0]*n for _ in range(m)]
    for i in range(n):
        tab[0][i]=i
    for i in range(m):
        tab[i][0]=i
    for i in range(1,m):
        for j in range(1,n):
            tab[i][j]=min(tab[i-1][j-1]+int(s1[i]!=s2[j]),tab[i][j-1]+1,tab[i-1][j]+1)
    return tab[-1][-1]



num_class=config.NUM_CLASS

max_label_length=config.MAX_LABEL_LENGTH

train_dir=config.TRAIN_DIR
valid_dir=config.VALID_DIR

check_dir=config.CHECKPOINT_DIR
check_path=os.path.join(check_dir,'024-loss:0.233-val_loss:2.540.h5')
char_map_dict =config.CHAR_MAP_DICT


dataset=load_data.TextData(valid_dir,char_map_dict,None,max_label_length,None)
indexs=dataset.__index__

infer=inference.Inference(phase='test',
                hidden_num=256,
                layers_num=20,
                num_classes=37,
                input_tensor_shape=(32,None,3),
                check_path=check_path,
                char_map_dict=char_map_dict)
total=0
for i in indexs:
    image_raw,_,groundtruth=dataset.__getitem__(i)
    _,pred,_=infer.predict(image_raw)
    print (i)
#    edit=editDistance(pred,groundtruth)
#    print (i,edit,groundtruth,pred)
#    total+=edit
#mean_edit=total/float(len(indexs))
