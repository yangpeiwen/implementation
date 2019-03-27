#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from network_construction import GAN
#输入本次训练的数据集，这里以mnist为例
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#测试所需要的为生成器的随机噪声
gen_input = tf.placeholder(tf.float32, shape=[None, GAN.noise_dim], name='input_noise')
#GAN_predict为生成器函数的输入预测函数，实例化
gen_sample = GAN.GAN_predict(gen_input)
n = 6
#载入之前训练好的参数
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'./tmp/test_model.ckpt')
    
    
    canvas = np.empty((28 * n, 28 * n))
    for i in range(n):
        #噪声输入
        z = np.random.uniform(-1., 1., size=[n, GAN.noise_dim])
        #喂入噪声，获取gen_sample
        g = sess.run(gen_sample, feed_dict={gen_input: z})
        g = -1 * (g - 1)
        for j in range(n):
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()

