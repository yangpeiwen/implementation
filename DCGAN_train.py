#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#D网络中使用LeakyReLU作为激活函数
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from network_construction import DCGAN
#载入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# 网路的两个输入，生成器需要的随机噪声和判别器需要的真实图片
noise_input = tf.placeholder(tf.float32, shape=[None, DCGAN.noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
#batch normalization需要的is_training参数
#is_training在这里需要还有DCGAN文件里面的函数也需要，直接调用DCGAN文件的placeholder创建
is_training = DCGAN.is_training
#训练网络实例化，返回的gen_vars与disc_vars暂时用不到
train_gen,train_disc,gen_loss,disc_loss,gen_vars,disc_vars = DCGAN.DCGAN_train(noise_input,real_image_input)
#开始训练
init = tf.global_variables_initializer()
sess = tf.Session()
#下面的训练参数与网络参数在DCGAN文件中都有，例：可以直接使用num_steps或者删去下面参数然后DCGAN.num_steps
# Training Params
num_steps = 10000
batch_size = 128
lr_generator = 0.002
lr_discriminator = 0.002

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
noise_dim = 100 # Noise data points
#初始化并且创建saver对象准备保存
sess.run(init)
saver = tf.train.Saver()
model_path = "/tmp/DCGAN_model.ckpt"

for i in range(1, DCGAN.num_steps+1):

    batch_x, _ = mnist.train.next_batch(DCGAN.batch_size)
    batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
    batch_x = batch_x * 2. - 1.

    # 训练判别器
    z = np.random.uniform(-1., 1., size=[DCGAN.batch_size, DCGAN.noise_dim])
    _, dl = sess.run([train_disc, disc_loss], feed_dict={real_image_input: batch_x, noise_input: z, is_training:True})
    
    # 训练生成器
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, gl = sess.run([train_gen, gen_loss], feed_dict={noise_input: z, is_training:True})
    
    if i % 500 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
        
save_path = saver.save(sess,model_path)
print("Model saved in file: %s" % save_path)

