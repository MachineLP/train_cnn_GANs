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
from keras.utils import np_utils

# inception_v4
try:
    from inception_v4 import inception_v4_arg_scope, inception_v4
except:
    from lib.model.generator.inception_v4.inception_v4 import inception_v4_arg_scope, inception_v4
# resnet_v2_50, resnet_v2_101, resnet_v2_152
try:
    from resnet_v2 import resnet_arg_scope, resnet_v2_50
except:
    from lib.model.generator.resnet_v2.resnet_v2 import resnet_arg_scope, resnet_v2_50
# vgg16, vgg19
try:
    from vgg import vgg_arg_scope, vgg_16, vgg_16_conv
except:
    from lib.model.generator.vgg.vgg import vgg_arg_scope, vgg_16, vgg_16_conv

try:
    from alexnet import alexnet_v2_arg_scope, alexnet_v2
except:
    from lib.model.generator.alexnet.alexnet import alexnet_v2_arg_scope, alexnet_v2

try:
    from lp_net import lp_net, lp_net_arg_scope
except:
    from lib.model.generator.lp_net.lp_net import lp_net, lp_net_arg_scope

try:
    from attention import attention
except:
    from lib.model.generator.attention.attention import attention


class generator_arch(object):
    
    def __init__(self):
        pass
    
    def generator(self, z):
        """Generate images from a random noise vector.
    
        Inputs:
        - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
        Returns:
        TensorFlow Tensor of generated images, with shape [batch_size, 784].
        """
        with tf.variable_scope("generator"):
            # TODO: implement architecture
            dense_1 = tf.layers.dense(inputs=z,units=1024,activation=tf.nn.relu,use_bias=True)
            dense_2 = tf.layers.dense(inputs=dense_1,units=1024,activation=tf.nn.relu,use_bias=True)
            dense_3 = tf.layers.dense(inputs=dense_2,units=784,use_bias=True)
            img=tf.tanh(dense_3)
        return img, dense_3

    def generator_conv(self, z):
        """Generate images from a random noise vector.
        
        Inputs:
        - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
        
        Returns:
        TensorFlow Tensor of generated images, with shape [batch_size, 784].
        """
        with tf.variable_scope("generator"):
            # TODO: implement architecture
            net = tf.layers.dense(inputs=z,units=1024,activation=tf.nn.relu,use_bias=True)
            net = tf.reshape(net, [-1, 16, 16, 4])
            
            net = tf.layers.conv2d(net, 16, 3, 1, 'same', use_bias=True, activation=None)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            
            net = tf.layers.conv2d(net, 16, 3, 1, 'same', use_bias=True, activation=None)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            
            net = tf.layers.conv2d(net, 3, 3, 1, 'same', use_bias=True, activation=None)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            
            img=tf.tanh(net)
        return img, dense_3









