#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 22:17:51 2018

@author: dirac
"""

import keras.backend as K
import tensorflow as tf
import keras
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import json
import os
import sys
sys.path.append('../')
import cv2
import numpy as np
from matplotlib import pyplot as plt


from utils import crnn_model
from utils import load_data
from crnn import config

#forbidden GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

def format_img(image_raw,h):
    height,width,_=image_raw.shape
    w=int((h*width)/height)
#    w=100
    
    image = cv2.resize(image_raw, (w, h))    
    image=np.expand_dims(image,axis=0)
#    image=image/255.
    return image,w

class Inference(crnn_model.CRNNCTCNetwork):
    def __init__(self, phase, hidden_num, layers_num, num_classes, input_tensor_shape,check_path,char_map_dict):
        super(Inference, self).__init__(phase, hidden_num, layers_num, num_classes, input_tensor_shape)
        self.model=self.build_network()
        self.model.load_weights(check_path, by_name=True)
        self.h=input_tensor_shape[0]
        self.char_map_dict=char_map_dict
        self.inverse_dict=dict(zip(char_map_dict.values(), char_map_dict.keys()))
        
    def predict(self,image_raw):
        image,w=format_img(image_raw,self.h)
        input_length=np.array([int(w/4)])
        
        y_pred=self.model.predict(image)
        y_pred_labels_tensor_list, prob = keras.backend.ctc_decode(y_pred, input_length, greedy=True) # 使用的是最简单的贪婪算法
        y_pred_labels_tensor = y_pred_labels_tensor_list[0]
        y_pred_labels = keras.backend.get_value(y_pred_labels_tensor)[0] # 现在还是字符编码
#        return y_pred_labels

        y_pred_txt=load_data.int2txt(y_pred_labels,self.inverse_dict)
        y_pred_prob = keras.backend.get_value(prob)
        y_pred_prob=np.exp(-y_pred_prob)[0][0]
        
        return y_pred_labels,y_pred_txt,y_pred_prob
        







if __name__=='__main__':
    data_dir='../data/synth90k/valid/671/3'
    file_list=os.listdir(data_dir)
    check_dir=config.CHECKPOINT_DIR
    check_path=os.path.join(check_dir,'024-loss:0.233-val_loss:2.540.h5')



    h=config.TRAIN_H
    
    char_map_dict = json.load(open('../data/char_map.json', 'r'))
       
    infer=Inference(phase='test',
                    hidden_num=256,
                    layers_num=20,
                    num_classes=37,
                    input_tensor_shape=(h,None,3),
                    check_path=check_path,
                    char_map_dict=char_map_dict)
    
    
    for i in range(90):
        print (i)
        image_path=os.path.join(data_dir,file_list[i])
    
      
        image_raw = cv2.imread(image_path)
        a=infer.predict(image_raw)
        
    #    fig=plt.figure(figsize=[w/8,h/8])
    #    plt.imshow(image_raw)
    #    plt.title('%s-%.3f'%(y_pred_txt,y_pred_prob[0][0]),fontsize=33)
    #    print (i,y_pred_txt,y_pred_prob)

