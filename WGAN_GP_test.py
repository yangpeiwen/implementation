#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from network_construction import WGAN_GP
#本次训练载入数据集，以mnist为例
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#占位符分别为要输入的两个数据，一个是生成器的输入噪声。一个是判别器输入的mnist数据集，noise_dim,image_dim在GAN中定义
#测试所需要的为生成器的随机噪声
gen_input = tf.placeholder(tf.float32, shape=[None, WGAN_GP.noise_dim], name='input_noise')
#WGAN_GP_predict为生成器函数的输入预测函数，实例化
gen_sample = WGAN_GP.WGAN_GP_predict(gen_input)
n = 6
#载入之前训练好的参数
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'./tmp/WGAN_GP_model.ckpt')
    
    
    canvas = np.empty((28 * n, 28 * n))
    for i in range(n):
        #噪声输入
        z = np.random.uniform(-1., 1., size=[n, WGAN_GP.noise_dim])
        #喂入噪声，获取gen_sample
        g = sess.run(gen_sample, feed_dict={gen_input: z})
        g = -1 * (g - 1)
        for j in range(n):
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()


# In[ ]:




