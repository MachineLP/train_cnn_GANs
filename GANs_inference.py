#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np
from scipy import misc
import tensorflow as tf
from threading import Lock
import os
import cv2
import sys
from lib.utils.GANs_utils import sample_noise
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import config

model_class = 1
dim = 64
num_gen = 128

def GPU_config(rate=0.99):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpuConfig = tf.ConfigProto()
    gpuConfig.allow_soft_placement = False
    gpuConfig.gpu_options.allow_growth = True
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = rate

    return gpuConfig

def prewhiten(self, img):
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0/np.sqrt(img.size))
    ret = np.multiply(np.subtract(img, mean), 1/std_adj)
    return ret
def to_rgb(self,img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
def img_crop(img, box):
    # y1, x1, y2, x2 = box[1]-20, box[0]-20, box[1]+box[3]+40, box[0]+box[2]+40
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img = img[y1:y2, x1:x2]
    return img
def data_norm(img):
    img = img / 255.0
    img = img - 0.5
    img = img * 2
    return img
def dec_data_norm(img):
    img = img / 2.
    img = img + 0.5
    img = img * 255.
    return img

class LPAlg_unconditional(object):

    # default model path of .pb
    PB_PATH_1 = os.path.join(os.getcwd(), "model", "body_pose_model.pb")
    PB_PATH = [PB_PATH_1]

    CLASS_NUMBER = model_class

    def __init__(self, pb_path_1=None, gpu_config=GPU_config()):
        def get_path(path,default_path):
            return (path, default_path)[path is None]

        def load_graph(frozen_graph_filename):
            # We load the protobuf file from the disk and parse it to retrieve the
            # unserialized graph_def
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Then, we can use again a convenient built-in function to import a graph_def into the
            # current default Graph
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="prefix",
                    op_dict=None,
                    producer_op_list=None
                )
            return graph

        # model
        def sess_def(pb_path):
            print (pb_path)
            graph = load_graph(pb_path)
            pred = graph.get_tensor_by_name('prefix/predictions:0')
            batch_size = tf.placeholder(tf.float32, [None, 1])
            label_indices = tf.placeholder(tf.float32, [None, 2])
            x1 = graph.get_tensor_by_name('prefix/inputs_placeholder:0')
            # x2 = graph.get_tensor_by_name('prefix/inputs_placeholder2:0')
            # x3 = graph.get_tensor_by_name('prefix/inputs_placeholder3:0')
            sess = tf.Session(graph=graph,config=gpu_config)
            return [sess,x1,batch_size,label_indices,pred]

        # multiple models
        def multi_model_def(pb_path_1):
            model_1 = sess_def(pb_path_1)
            return [model_1]

        path_1 = get_path(pb_path_1, LPAlg_unconditional.PB_PATH[0])
        self._pb_path = [path_1]

        self.model = multi_model_def(self._pb_path[0])

    def _close(self):
        self.model[0][0].close()

    def _run(self, images_path=None):
        idx = 0     # model index
        # print (imgs)
        sess = tf.Session()
        generator_x = sess.run(sample_noise(num_gen, dim))
        predict = self.model[idx][0].run(
            self.model[idx][4],
            feed_dict={self.model[idx][1]: generator_x
                       # self.model[idx][2]: imgs2,
                       # self.model[idx][3]: imgs3
                                  }
                                  )
        print ('predict:', predict)

        return predict.tolist()

class LPAlg_conditional(object):

    # default model path of .pb
    PB_PATH_1 = os.path.join(os.getcwd(), "model", "body_pose_model.pb")
    PB_PATH = [PB_PATH_1]

    CLASS_NUMBER = model_class

    def __init__(self, pb_path_1=None, gpu_config=GPU_config()):
        def get_path(path,default_path):
            return (path, default_path)[path is None]

        def load_graph(frozen_graph_filename):
            # We load the protobuf file from the disk and parse it to retrieve the
            # unserialized graph_def
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Then, we can use again a convenient built-in function to import a graph_def into the
            # current default Graph
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="prefix",
                    op_dict=None,
                    producer_op_list=None
                )
            return graph

        # model
        def sess_def(pb_path):
            print (pb_path)
            graph = load_graph(pb_path)
            pred = graph.get_tensor_by_name('prefix/predictions:0')
            batch_size = tf.placeholder(tf.float32, [None, 1])
            label_indices = tf.placeholder(tf.float32, [None, 2])
            x1 = graph.get_tensor_by_name('prefix/inputs_placeholder0:0')
            x2 = graph.get_tensor_by_name('prefix/inputs_placeholder1:0')
            # x2 = graph.get_tensor_by_name('prefix/inputs_placeholder2:0')
            # x3 = graph.get_tensor_by_name('prefix/inputs_placeholder3:0')
            sess = tf.Session(graph=graph,config=gpu_config)
            return [sess,x1,x2,batch_size,label_indices,pred]

        # multiple models
        def multi_model_def(pb_path_1):
            model_1 = sess_def(pb_path_1)
            return [model_1]

        path_1 = get_path(pb_path_1, LPAlg_conditional.PB_PATH[0])
        self._pb_path = [path_1]

        self.model = multi_model_def(self._pb_path[0])

    def _close(self):
        self.model[0][0].close()

    def _run(self, images_path=None):
        idx = 0     # model index
        # print (imgs)
        sess = tf.Session()
        generator_x = sess.run(sample_noise(num_gen, dim))
        num_classes = config.num_classes
        label = np.zeros([num_classes])
        label[0] = 1
        predict = self.model[idx][0].run(
            self.model[idx][5],
            feed_dict={self.model[idx][1]: generator_x,
                       self.model[idx][2]: [label]
                       # self.model[idx][3]: imgs3
                                  }
                                  )
        print ('predict:', predict)

        return predict.tolist()

def ProjectInterface(image_path_list, proxy=None):
    images_path = image_path_list.keys()
    predict = proxy._run(images_path)
    return predict



def show_images(images):
    # images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    # sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    # sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    
    fig = plt.figure(figsize=(36, 36))
    gs = gridspec.GridSpec(36, 36)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, img in enumerate(images):
        img = np.asarray(img)
        img = dec_data_norm(img)
        cv2.imwrite('face0.jpg', img)
        img = cv2.imread('face0.jpg')
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # plt.imshow(img.reshape([sqrtimg,sqrtimg]))
        plt.imshow(img.reshape([32,32,3]))


if __name__ == "__main__":
    # python predict.py lp.jpg (带标签输出逻辑)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Assign the image path.', default="")
    args = parser.parse_args()
    arch_model = config.arch_model
    if arch_model == "arch_dcgan_unconditional":
        alg_core = LPAlg_unconditional(pb_path_1="model/body_pose_model.pb")
    elif arch_model == "arch_dcgan_conditional":
        alg_core = LPAlg_conditional(pb_path_1="model/body_pose_model.pb")
    else:
        print ('{} is error!', arch_model)
    result_dict = ProjectInterface({args.image: args.image}, proxy=alg_core)
    result_dict_img = ((np.asarray(result_dict) / 2. + 0.5) *255) #.reshape([32,32,3])
    print(result_dict_img)
    cv2.imwrite('face.jpg', result_dict_img[0])
    result_dict = np.asarray(result_dict)
    show_images(result_dict)
    plt.show()
