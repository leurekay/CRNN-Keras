#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:43:49 2018

@author: dirac
"""

import os
import cv2
import numpy as np
import json


def txt2int(txt,map_dict,max_len):
    box=[]
    for k in txt:
        box.append(map_dict[k])
    n=len(box)
    box.extend([-1]*max_len)
    return box[:max_len]

def int2txt(nums,inverse_map_dict):
    txt=''
    for num in nums:
        char=inverse_map_dict[num]
        txt+=char
    return txt



class TextData(object):
    def __init__(self,image_dir,map_dict,resize_shape,max_label_len,input_len):
        self.map_dict=map_dict
        self.resize_shape=resize_shape
        self.max_label_len=max_label_len
        self.input_len=input_len
        walk=os.walk(image_dir)
        box=[]
        for root, dirs, files in walk:
            for name in files:
                if name.endswith('jpg'):
                    box.append(os.path.join(root,name))
        self.file_list=box
        self.__nums__=len(box)
        self.__index__=np.array(range(len(box)))
    def __shuffle(self):
        ind=np.array(range(len(self.file_list)))
        np.random.shuffle(ind)
        self.file_list=list(map(lambda i : self.file_list[i],ind))

    
    def __raw_item__(self,index):
        assert index<self.__nums__
        image_path=self.file_list[index]
        image = cv2.imread(image_path)
     
        _,filename=os.path.split(image_path)
        name,_=os.path.splitext(filename)
        txt=name.split('_')[1].lower()
        return image,txt        
    
    def __getitem__(self,index):
        assert index<self.__nums__
        image,txt=self.__raw_item__(index)
#        image=image/255.
        
        if self.resize_shape:
            width,height=self.resize_shape
            try:
                image = cv2.resize(image, (width, height))    
            except:
                print (index,txt)
                return -99,[-99],-99
            
        label=txt2int(txt,self.map_dict,self.max_label_len)
#        print (index,txt)
        return image,label,txt
    
    
    def get_batch(self,indexs):
        box_images=[]
        box_labels=[]
        box_length=[]
        for index in indexs:
            image,label,txt=self.__getitem__(index)
            if label[0]==-99:
                indexs.append(np.random.choice(self.__index__))
                continue
            box_images.append(image)
            box_labels.append(label)
            box_length.append(len(txt))
        box_length=np.array(box_length)
        input_length=self.input_len*np.ones(box_length.shape)
        return np.array(box_images),np.array(box_labels),input_length,box_length
    
    
    def next_batch(self,batch_size):
        start=0
        while True:
#            end=min(self.__nums__,start+batch_size)
            end=start+batch_size
            indexs=list(range(start,end))
            images,labels,input_length,label_lengths=self.get_batch(indexs)
            
            if end==self.__nums__-self.__nums__%batch_size:
                start=0
                self.__shuffle()
            else:
                start+=batch_size


            inputs = {
                'the_input': images,  # (bs, 128, 64, 1)
                'the_labels': labels,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_lengths  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)
            
#            yield(images,labels,label_lengths)
     


if __name__=='__main__':
    
    data_dir='../data/synth90k'
    list_path=os.path.join(data_dir,'image_list.txt')
    char_map_dict = json.load(open('../data/char_map.json', 'r'))

    
    txtdata=TextData(data_dir,char_map_dict,(100,32),9,25)
    ll=txtdata.file_list
    
    image=txtdata.__getitem__(432)
    
    ooxx=txtdata.get_batch([3,45,23,8])
    
    box=[]
#    for i in txtdata.next_batch(32):
#        box.append(i)
    
    img=cv2.imread('../data/synth90k/2911/6/77_heretical_35885.jpg')
    
    
#    list_path=os.path.join(data_dir,'image_list.txt')
#    
#    with open(list_path) as f:
#        txt=f.readlines()
#        txt=list(map(lambda x : x[:-1],txt))
#        
#    image_path=os.path.join(data_dir,txt[-1])
#    #image_path='../data/synth90k/701/4/242_Snapdragon_72126.jpg'
#    image = cv2.imread(image_path)