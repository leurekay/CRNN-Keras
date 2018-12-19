#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:06:08 2018

@author: ly
"""

import keras.backend as K
import tensorflow as tf

x=K.constant(1,shape=[10,2])
mask_pos=tf.greater(x,-0.5)
mask_int=tf.cast(mask_pos,tf.int16)
length=tf.reduce_sum(mask_int,axis=1)

#y=tf.get_variable(shape=[32,8],dtype=tf.int32,name='yy',initializer=tf.random_normal_initializer(mean=0, stddev=10))
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    out = sess.run(x)
    mask=sess.run(mask_pos)
    oo=sess.run(mask_int)
    ll=sess.run(length)

