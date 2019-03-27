#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from network_construction import WGAN_GP
#from network_construction import GAN
#本次训练载入数据集，以mnist为例
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#占位符分别为要输入的两个数据，一个是生成器的输入噪声。一个是判别器输入的mnist数据集，noise_dim,image_dim在GAN中定义
gen_input = tf.placeholder(tf.float32, shape=[None, WGAN_GP.noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, WGAN_GP.image_dim], name='disc_input')
#实例化训练参数
train_gen,train_disc,gen_loss,disc_loss,gen_vars, disc_vars = WGAN_GP.WGAN_GP_network(gen_input,disc_input)
#开始训练
sess = tf.Session()
init = tf.global_variables_initializer()
# 初始化
sess.run(init)
#创建saver对象用来保存训练好的参数,model_path是模型文件的保存路径
saver = tf.train.Saver()
model_path = "./tmp/WGAN_GP_model.ckpt"

for i in range(1, WGAN_GP.num_steps+1):
    #这里以mnist为例子
    batch_x, _ = mnist.train.next_batch(WGAN_GP.batch_size)
    #给生成器输入噪声,-1到1之间符合正态分布
    z = np.random.uniform(-1., 1., size=[WGAN_GP.batch_size, WGAN_GP.noise_dim])

    # Train，喂入一个batch的图片矩阵和生成器输入噪声
    feed_dict = {disc_input: batch_x, gen_input: z}
    _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                            feed_dict=feed_dict)
    if i % 2000 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
        
save_path = saver.save(sess,model_path)
print("Model saved in file: %s" % save_path)


# In[ ]:




