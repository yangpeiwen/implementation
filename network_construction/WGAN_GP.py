#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
#GAN的参数与网络结构
#训练参数
num_steps = 70000
batch_size = 128
learning_rate = 0.0002

#网络参数
image_dim = 784 # 28x28的mnist图片像素
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # 噪声结构

#Xavier初始化
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

#生成器与判别器的参数变量
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1])),
}

#生成器与判别器的定义
#生成器
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


#判别器
def discriminator(x):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    #WGAN去掉判别器的sigmoid
    #out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

def WGAN_GP_predict(gen_input):
    gen_sample = generator(gen_input)
    return gen_sample

#GAN网络结构
def WGAN_GP_network(input_noise, disc_input):

    gen_input = input_noise

    #建立生成器网络
    gen_sample = generator(gen_input)

    #disc_real是给判别器输入真实的图片,disc_fake是给判别器输入生成器生成的图片
    disc_real = discriminator(disc_input)
    disc_fake = discriminator(gen_sample)
    
    #WGAN的gradient penalty,设置一个额外的loss项来添加到disc_loss后面
    #先随机采取一对真假样本，还有一个0-1的随机数
    #eps是0-1之间随机数
    eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
    #在真实样本和生成样本之间随机插入一个值
    X_inter = eps*disc_input + (1. - eps)*gen_sample
    #求梯度
    grad = tf.gradients(discriminator(X_inter), [X_inter])[0]
    #求梯度的二范数
    grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
    #Lipschitz限制的k定为1
    grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))
        
    #WGAN的loss函数，判别器加入了grad_pen
    gen_loss = -tf.reduce_mean(disc_fake)
    disc_loss = -tf.reduce_mean(disc_real)+tf.reduce_mean(disc_fake)+grad_pen

    #adam优化器，事实上WGAN作者不建议使用基于向量的adam的优化器，所以本实验生成结果不佳，考虑调节其他优化器 
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    #  Training Variables for each optimizer
    # By default in TensorFlow, all variables are updated by each optimizer, so we
    # need to precise for each one of them the specific variables to update.
    # Generator Network Variables
    gen_vars = [weights['gen_hidden1'], weights['gen_out'],
                biases['gen_hidden1'], biases['gen_out']]
    # Discriminator Network Variables
    disc_vars = [weights['disc_hidden1'], weights['disc_out'],
                biases['disc_hidden1'], biases['disc_out']]

    # Create training operations
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    # Initialize the variables (i.e. assign their default value)
    #init = tf.global_variables_initializer()
    return train_gen,train_disc,gen_loss,disc_loss,gen_vars,disc_vars

