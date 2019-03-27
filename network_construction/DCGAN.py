#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#DCGAN对卷积神经网络的结构做了一些改变，以提高样本的质量和收敛的速度，这些改变有：
#取消所有pooling层。G网络中使用转置卷积（transposed convolutional layer）进行上采样，D网络中用加入stride的卷积代替pooling。
#在D和G中均使用batch normalization（让数据更集中，不用担心太大或者太小的数据，可以稳定学习，有助于处理初始化不良导致的训练问题，也有助于梯度流向更深的网络，防止G崩溃。同时，让学习效率变得更高。)
#去掉全连接层，而直接使用卷积层连接生成器和判别器的输入层以及输出层，使网络变为全卷积网络
#G网络中使用ReLU作为激活函数，最后一层使用tanh
#D网络中使用LeakyReLU作为激活函数
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#载入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Training Params
num_steps = 10000
batch_size = 128
lr_generator = 0.002
lr_discriminator = 0.002

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
noise_dim = 100 # Noise data points


# Build Networks
# Network Inputs
#noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
#real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# A boolean to indicate batch normalization if it is training or inference time
is_training = tf.placeholder(tf.bool)

#LeakyReLU activation
def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

# 生成器网络
# Input: Noise, Output: Image
# Note that batch normalization has different behavior at training and inference time,
# we then use a placeholder to indicates the layer if we are training or not.

def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # 使用tensorflow函数来自动生成相关weight和bais
        x = tf.layers.dense(x, units=7 * 7 * 128)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        #取消所有pooling层。G网络中使用转置卷积（transposed convolutional layer）进行上采样
        # 反卷积以及矩阵格式: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')
        #在D和G中均使用batch normalization
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # 反卷积以及矩阵格式: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding='same')
        # G网络中使用ReLU作为激活函数，最后一层使用tanh
        x = tf.nn.tanh(x)
        return x


# 判别器网络
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        # Flatten
        x = tf.reshape(x, shape=[-1, 7*7*128])
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        #判别结果假或真
        x = tf.layers.dense(x, 2)
    return x

#构建生成网络
#gen_sample = generator(noise_input)
def DCGAN_train(noise_input,real_image_input):
    
    
    #构建生成网络
    gen_sample = generator(noise_input)
    #构建判别器网络，分别输入噪声与生成器的结果
    disc_real = discriminator(real_image_input)
    disc_fake = discriminator(gen_sample, reuse=True)

    # Build the stacked generator/discriminato
    stacked_gan = discriminator(gen_sample, reuse=True)

    #判别器损失函数(判别为真实图片: 1,判别为虚假图片: 0)
    # 构建判别器损失
    disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))
    #判别器损失之和
    disc_loss = disc_loss_real + disc_loss_fake
    #生成器损失函数(生成器需要能够通过判别器的认可, 标签为1)
    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=stacked_gan, labels=tf.ones([batch_size], dtype=tf.int32)))

    # 使用Adam优化器
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator, beta1=0.5, beta2=0.999)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)

    # Training Variables for each optimizer
    # By default in TensorFlow, all variables are updated by each optimizer, so we
    # need to precise for each one of them the specific variables to update.
    # Generator Network Variables
    #获取TRAINABLE_VARIABLES参数合集（即为所有变量）从Generator命名域中取得（Generator命名域中的是tf函数自建的weight和bias等）
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    #同上
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    # 建立训练参数
    #使用tensorflow的函数进行batch_norm正则化，需要更新所有的moving mean与stddev
     # TensorFlow UPDATE_OPS collection holds all batch norm operation to update the moving mean/stddev
    #UPDATE_OPS是一个tf内置的集合，train之前应当完成的操作，而batch_norm刚好就是需要在之前进行计算moving mean和stddy，这两部分已经加入了集合
    gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
    # `control_dependencies` ensure that the `gen_update_ops` will be run before the `minimize` op (backprop)
    #保证了在对生成器损失函数优化之前先计算moving mean与stddy
    with tf.control_dependencies(gen_update_ops):
        train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
    with tf.control_dependencies(disc_update_ops):
        train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)
    return train_gen,train_disc,gen_loss,disc_loss,gen_vars,disc_vars
# Initialize the variables (i.e. assign their default value)
#init = tf.global_variables_initializer()
#DCGAN根据原文，既可以像普通GAN那样用训练好的生成器，也可以用训练好的判别作为使用工具，这里以mnist的生成为例
def DCGAN_predict(noise_input):
    gen_sample = generator(noise_input)
    return gen_sample

