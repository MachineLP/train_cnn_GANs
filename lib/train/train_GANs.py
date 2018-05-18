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
from lib.utils.GANs_utils import build_generator, build_discriminator, cost, train_op, get_next_batch_from_path, gan_loss, preprocess_img
from lib.utils.GANs_utils import sample_noise, generator_input_placeholder, discriminator_input_placeholder,shuffle_train_data
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('./datasets/MNIST_data', one_hot=False)

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# A bunch of utility functions

def show_images(images):
    # images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    # sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    # sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    
    fig = plt.figure(figsize=(32, 32))
    gs = gridspec.GridSpec(32, 32)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # plt.imshow(img.reshape([sqrtimg,sqrtimg]))
        plt.imshow(img.reshape([32,32,3]))
    return 

# GPU 参数配置
def GPU_config(rate=0.8):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # 设置当前使用的GPU设备仅为0号设备
    gpuConfig = tf.ConfigProto()
    gpuConfig.allow_soft_placement = False      #设置为True，当GPU不存在或者程序中出现GPU不能运行的代码时，自动切换到CPU运行
    gpuConfig.gpu_options.allow_growth = True       #设置为True，程序运行时，会根据程序所需GPU显存情况，分配最小的资源
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = rate    #程序运行的时，所需的GPU显存资源最大不允许超过rate的设定值
    
    return gpuConfig

def train_GANs(train_data,train_label,valid_data,valid_label,train_dir,num_classes,batch_size,arch_model,learning_r_decay,learning_rate_base,decay_rate,dropout_prob,epoch,height,width,checkpoint_exclude_scopes,early_stop,EARLY_STOP_PATIENCE,fine_tune,train_all_layers,checkpoint_path,train_n,valid_n,g_parameter,dim = 64):
    # ---------------------------------------------------------------------------------#
    G_X, G_Y, G_is_train, G_keep_prob_fc = generator_input_placeholder(dim, num_classes)
    if arch_model == "arch_dcgan_unconditional":
        with tf.variable_scope("Generator_dcgan", reuse=None) as scope:
            G_net, _ = build_generator(G_X, num_classes, G_keep_prob_fc, G_is_train,arch_model)
    elif arch_model == "arch_dcgan_conditional":
        with tf.variable_scope("Generator_dcgan", reuse=None) as scope:
            G_net, _ = build_generator([G_X, G_Y], num_classes, G_keep_prob_fc, G_is_train,arch_model)
    else:
        print ('{} is error!', arch_model)

    D_X, D_Y, D_is_train, D_keep_prob_fc = discriminator_input_placeholder(height, width, num_classes)
    if arch_model == "arch_dcgan_unconditional":
        with tf.variable_scope("Discriminator_dcgan", reuse=tf.AUTO_REUSE) as scope:
            logits_real, _ = build_discriminator(D_X, num_classes, D_keep_prob_fc, D_is_train,arch_model)
            # scope.reuse_variables()
            logits_fake, _ = build_discriminator(G_net, num_classes, D_keep_prob_fc, D_is_train,arch_model)
    elif arch_model == "arch_dcgan_conditional":
        with tf.variable_scope("Discriminator_dcgan", reuse=tf.AUTO_REUSE) as scope:
            logits_real, _ = build_discriminator([D_X, D_Y], num_classes, D_keep_prob_fc, D_is_train,arch_model)
            # scope.reuse_variables()
            logits_fake, _ = build_discriminator([G_net,G_Y] , num_classes, D_keep_prob_fc, D_is_train,arch_model)
    else:
        print ('{} is error!', arch_model)

    #  tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator_dcgan')
    # print (G_vars)
    D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Discriminator_dcgan')
    # print (G_vars)

    D_loss, G_loss = gan_loss(logits_real, logits_fake)
    # G_loss = cost(logits_fake)
    # D_loss = cost(logits_real) + cost(logits_fake)

    global_step = tf.Variable(0, trainable=False)  
    if learning_r_decay:
        learning_rate = tf.train.exponential_decay(  
            learning_rate_base,                     
            global_step * batch_size,  
            train_n,                
            decay_rate,                       
            staircase=True)  
    else:
        learning_rate = learning_rate_base
    if arch_model == "arch_dcgan_unconditional":
        G_optimizer = train_op(learning_rate*10, G_loss, G_vars, global_step)
        D_optimizer = train_op(learning_rate, D_loss, D_vars, global_step)
    elif arch_model == "arch_dcgan_conditional":
        G_optimizer = train_op(learning_rate*10, G_loss, G_vars, global_step)
        D_optimizer = train_op(learning_rate, D_loss, D_vars, global_step)
    else:
        print ('{} is error!', arch_model)

    #------------------------------------------------------------------------------------#
    sess = tf.Session(config=GPU_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    saver2 = tf.train.Saver(G_vars)
    if not train_all_layers:
        saver_net = tf.train.Saver(G_vars)
        saver_net.restore(sess, checkpoint_path)
    
    if fine_tune:
        # saver2.restore(sess, fine_tune_dir)
        latest = tf.train.latest_checkpoint(train_dir)
        if not latest:
            print ("No checkpoint to continue from in", train_dir)
            sys.exit(1)
        print ("resume", latest)
        saver2.restore(sess, latest)
    
    # early stopping
    best_valid = np.inf
    best_valid_epoch = 0

    for epoch_i in range(epoch):
        for batch_i in range(int(train_n/batch_size)):
            dim = dim
            generator_x = sess.run(sample_noise(batch_size, dim))
            # images,_ = mnist.train.next_batch(batch_size)
            # images = preprocess_img(images)
            images, labels = get_next_batch_from_path(train_data, train_label, batch_i, height, width, batch_size=batch_size, training=True)
            if arch_model == "arch_dcgan_unconditional":
                D_los, _ = sess.run([D_loss,D_optimizer], feed_dict={G_X: generator_x, G_is_train:True, G_keep_prob_fc:dropout_prob,D_X: images,D_is_train:True, D_keep_prob_fc:dropout_prob})
                G_los, _ = sess.run([G_loss,G_optimizer], feed_dict={G_X: generator_x,G_is_train:True, G_keep_prob_fc:dropout_prob,D_X: images,D_is_train:True, D_keep_prob_fc:dropout_prob})
            elif arch_model == "arch_dcgan_conditional":
                D_los, _ = sess.run([D_loss,D_optimizer], feed_dict={G_X: generator_x,G_Y:labels,G_is_train:True,G_keep_prob_fc:dropout_prob,D_X: images,D_Y:labels,D_is_train:True, D_keep_prob_fc:dropout_prob})
                G_los, _ = sess.run([G_loss,G_optimizer], feed_dict={G_X: generator_x,G_Y:labels,G_is_train:True, G_keep_prob_fc:dropout_prob,D_X: images,D_is_train:True, D_keep_prob_fc:dropout_prob})
            else:
                print ('{} is error!', arch_model)
            print ('D_los:', D_los)
            print ('G_los:', G_los)
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver2.save(sess, checkpoint_path, global_step=batch_i, write_meta_graph=False)
            if batch_i%20==0:
                if arch_model == "arch_dcgan_unconditional":
                    D_loss_ = sess.run(D_loss, feed_dict={G_X: generator_x, G_is_train:True, G_keep_prob_fc:1.0,D_X: images,  D_X: images,D_is_train:False, D_keep_prob_fc:1.0})
                    G_loss_ = sess.run(G_loss, feed_dict={G_X: generator_x,  G_is_train:True, G_keep_prob_fc:1.0, D_X: images,D_is_train:False, D_keep_prob_fc:1.0})
                elif arch_model == "arch_dcgan_conditional":
                    D_loss_ = sess.run(D_loss, feed_dict={G_X: generator_x,G_Y:labels,G_is_train:True,G_keep_prob_fc:1.0,D_X: images,D_Y:labels,D_is_train:False, D_keep_prob_fc:1.0})
                    G_loss_ = sess.run(G_loss, feed_dict={G_X: generator_x,G_Y:labels,G_is_train:True, G_keep_prob_fc:1.0, D_X: images,D_is_train:False, D_keep_prob_fc:1.0})
                else:
                    print ('{} is error!', arch_model)

                print('Batch: {:>2}: D_training loss: {:>3.5f}'.format(batch_i, D_loss_))
                print('Batch: {:>2}: G_training loss: {:>3.5f}'.format(batch_i, G_loss_))

            if batch_i%100==0:
                generator_x = sess.run(sample_noise(batch_size, dim))
                # images,_ = mnist.train.next_batch(batch_size)
                # images = preprocess_img(images)
                images, labels = get_next_batch_from_path(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), height, width, batch_size=batch_size, training=False)
                if arch_model == "arch_dcgan_unconditional":
                    D_ls = sess.run(D_loss, feed_dict={G_X: generator_x, G_is_train:False, G_keep_prob_fc:1.0, D_X: images, D_is_train:False, D_keep_prob_fc:1.0})
                    G_ls = sess.run(G_loss, feed_dict={G_X: generator_x, G_is_train:False, G_keep_prob_fc:1.0, D_X: images,D_is_train:False, D_keep_prob_fc:1.0})
                elif arch_model == "arch_dcgan_conditional":
                    D_ls = sess.run(D_loss, feed_dict={G_X: generator_x,G_Y:labels,G_is_train:False,G_keep_prob_fc:1.0,D_X: images,D_Y:labels,D_is_train:False, D_keep_prob_fc:1.0})
                    G_ls = sess.run(G_loss, feed_dict={G_X: generator_x,G_Y:labels,G_is_train:False, G_keep_prob_fc:1.0, D_X: images,D_is_train:False, D_keep_prob_fc:1.0})
                else:
                    print ('{} is error!', arch_model)

                print('Batch: {:>2}: D_validation loss: {:>3.5f}'.format(batch_i, D_ls))
                print('Batch: {:>2}: G_validation loss: {:>3.5f}'.format(batch_i, G_ls))
            
        
        print('Epoch===================================>: {:>2}'.format(epoch_i))
        G_valid_ls = 0
        G_samples = []
        for batch_i in range(int(valid_n/batch_size)):
            generator_x = sess.run(sample_noise(batch_size, dim))
            if arch_model == "arch_dcgan_unconditional":
                images, labels = get_next_batch_from_path(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), height, width, batch_size=batch_size, training=False)
                G_epoch_ls, G_samples = sess.run([G_loss, G_net], feed_dict={G_X: generator_x, G_keep_prob_fc:1.0, G_is_train:False, D_X: images,D_is_train:False, D_keep_prob_fc:1.0})
            elif arch_model == "arch_dcgan_conditional":
                images, labels = get_next_batch_from_path(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), height, width, batch_size=batch_size, training=False)
                G_epoch_ls, G_samples = sess.run([G_loss, G_net], feed_dict={G_X: generator_x, G_Y:labels, G_keep_prob_fc:1.0, G_is_train:False, D_X: images,D_is_train:False, D_keep_prob_fc:1.0})

            G_valid_ls = G_valid_ls + G_epoch_ls
        fig = show_images(G_samples[:16])
        plt.show()
        print('Epoch: {:>2}: G_validation loss: {:>3.5f}'.format(epoch_i, G_valid_ls/int(valid_n/batch_size)))
        # ---------------------------------------------------------------------------------#
        if early_stop:
            loss_valid = G_valid_ls/int(valid_n/batch_size)
            if loss_valid < best_valid:
                best_valid = loss_valid
                best_valid_epoch = epoch_i
            elif best_valid_epoch + EARLY_STOP_PATIENCE < epoch_i:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch))
                break
        train_data, train_label = shuffle_train_data(train_data, train_label)
    sess.close()
