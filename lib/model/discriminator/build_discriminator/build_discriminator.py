# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import argparse
import os
from PIL import Image
from datetime import datetime
import math
import time
import cv2
# from lib.utils.gans_utils import leaky_relu

def fully_connected(prev_layer, num_units, is_training):
    
    layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


def conv_layer(prev_layer, layer_depth, is_training):
    
    strides = 2 if layer_depth % 3 == 0 else 1
    conv_layer = tf.layers.conv2d(prev_layer, layer_depth*16, 3, strides, 'same', use_bias=True, activation=None)
    conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    conv_layer = tf.nn.relu(conv_layer)
    
    return conv_layer

def unpool(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ResidualConvUnit(inputs,features=256,kernel_size=3):
    net=tf.nn.relu(inputs)
    net=slim.conv2d(net, features, kernel_size)
    net=tf.nn.relu(net)
    net=slim.conv2d(net,features,kernel_size)
    net=tf.add(net,inputs)
    return net
def MultiResolutionFusion(high_inputs=None,low_inputs=None,features=256):
    
    if high_inputs is None:#refineNet block 4
        rcu_low_1 = low_inputs[0]
        rcu_low_2 = low_inputs[1]
        
        rcu_low_1 = slim.conv2d(rcu_low_1, features, 3)
        rcu_low_2 = slim.conv2d(rcu_low_2, features, 3)
        
        return tf.add(rcu_low_1,rcu_low_2)

class discriminator_arch(object):
    def __inint__(self):
        pass

    def discriminator(self, x):
        """Compute discriminator score for a batch of input images.
    
        Inputs:
        - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score 
        for an image being real for each input image.
        """
        with tf.variable_scope("discriminator"):
            # TODO: implement architecture
            dense_1 = tf.layers.dense(inputs=x,units=256,use_bias=True)
            # relu_1=leaky_relu(dense_1, alpha=0.01)
            relu_1=tf.nn.relu(dense_1)
            dense_2 = tf.layers.dense(inputs=relu_1,units=256,use_bias=True)
            # relu_2=leaky_relu(dense_2, alpha=0.01)
            relu_2 = tf.nn.relu(dense_2)
            dense_3 = tf.layers.dense(inputs=relu_2,units=1,use_bias=True)
        
            logits=dense_3
            return logits, dense_3

    def discriminator_conv(self, x):
        """Compute discriminator score for a batch of input images.
        
        Inputs:
        - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
        
        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score
        for an image being real for each input image.
        """
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope("discriminator"):
                for layer_i in [1,2,4]:
                    for n in range(1):
                        layer = conv_layer(layer, layer_i, is_training)
                    layer = tf.nn.max_pool(layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
                net = unpool(layer, 2)
                net = tf.layers.conv2d(net, 256, 3, 1, 'SAME', use_bias=True, activation=None)
                net = ResidualConvUnit(net, 256)
                net=tf.nn.relu(net)
            
                net = unpool(layer, 2)
                net = tf.layers.conv2d(net, 128, 3, 1, 'SAME', use_bias=True, activation=None)
                net = ResidualConvUnit(net, 128)
                net=tf.nn.relu(net)

                net = unpool(layer, 2)
                net = tf.layers.conv2d(net, 64, 3, 1, 'SAME', use_bias=True, activation=None)
                net = ResidualConvUnit(net, 64)
                net=tf.nn.relu(net)
            
                net = tf.layers.conv2d(net, num_classes, 3, 1, 'SAME', use_bias=True, activation=None)
            
            return net, layer


