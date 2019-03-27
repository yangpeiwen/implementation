#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from network_construction import DCGAN
#输入本次训练的数据集，这里以mnist为例
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#网络参数，也可以去掉下面的，然后用DCGAN.前缀使用
image_dim = 784 # 28*28 pixels * 1 channel
noise_dim = 100 # Noise data points

#测试所需要的为生成器的随机噪声
gen_input = tf.placeholder(tf.float32, shape=[None, DCGAN.noise_dim], name='input_noise')
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
is_training = DCGAN.is_training
#GAN_predict为生成器函数的输入预测函数，实例化
gen_sample = DCGAN.DCGAN_predict(gen_input)

batch_x, _ = mnist.train.next_batch(DCGAN.batch_size)
batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
batch_x = batch_x * 2. - 1.

#开始测试
n = 6
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'./tmp/DCGAN_model.ckpt')
    
    
    canvas = np.empty((28 * n, 28 * n))
    for i in range(n):
        #噪声输入
        z = np.random.uniform(-1., 1., size=[n, DCGAN.noise_dim])
       #测试生成器
        g = sess.run(gen_sample, feed_dict={real_image_input: batch_x, gen_input: z, is_training:True})        
        g = (g + 1.) / 2.
        g = -1 * (g - 1)
        for j in range(n):
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()

