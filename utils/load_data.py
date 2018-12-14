#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:43:49 2018

@author: dirac
"""

import os
import cv2
import numpy as np




class TextData(object):
    def __init__(self,image_dir):
        walk=os.walk(image_dir)
        box=[]
        for root, dirs, files in walk:
            for name in files:
                if name.endswith('jpg'):
                    box.append(os.path.join(root,name))
        self.file_list=box
        self.len=len(box)
        self.index=np.array(range(len(box)))
    def shuffle(self):
        ind=np.array(range(len(self.file_list)))
        np.random.shuffle(ind)
        self.file_list=list(map(lambda i : self.file_list[i],ind))

    def getitem(self,ind):
        assert ind<self.len
        image_path=self.file_list[ind]
        image = cv2.imread(image_path)
        
        _,filename=os.path.split(image_path)
        name,_=os.path.splitext(filename)
        label=name.split('_')[1]
        
        return image,label
    
    
    def get_batch(self,indexes,new_shape):
        pass


if __name__=='__main__':
    
    data_dir='../data/synth90k'
    list_path=os.path.join(data_dir,'image_list.txt')
    
    txtdata=TextData(data_dir)
    ll=txtdata.file_list
    
#    list_path=os.path.join(data_dir,'image_list.txt')
#    
#    with open(list_path) as f:
#        txt=f.readlines()
#        txt=list(map(lambda x : x[:-1],txt))
#        
#    image_path=os.path.join(data_dir,txt[-1])
#    #image_path='../data/synth90k/701/4/242_Snapdragon_72126.jpg'
#    image = cv2.imread(image_path)