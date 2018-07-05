#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 22:16:55 2018

@author: wsw
"""

# read tfrecords

import tensorflow as tf
import os
import matplotlib.pyplot as plt

tf.reset_default_graph()

datasetPath = './mnist'

# 解析一个训练样例
def parser_example(serialized_example):
    
    # features{}key必须和制作tfrecords的key一致
    # 且value的数据类型要一致
    features = tf.parse_single_example(serialized_example,
                                       features={'feature':tf.FixedLenFeature([],tf.string),
                                                 'label':tf.FixedLenFeature([],tf.float32)})
    # 解码字节型(Bytes)张量
    image = tf.decode_raw(features['feature'],tf.uint8)
    image = tf.reshape(image,shape=[28,28])
    label = features['label']
    return image,label



def read_tfrecords(dataType='train'):
    
    file = os.path.join(datasetPath,'%s.tfrecords'%dataType)
    reader = tf.TFRecordReader()
    # 将tfrecords文件添加到文件名队列中
    filename_queue = tf.train.string_input_producer([file])
    # 读取得到一个序列化样例
    _,serialized_example = reader.read(filename_queue)
    # 解析一个序列化样例
    image,label = parser_example(serialized_example)
    
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            img,lab = sess.run([image,label])
            plt.imshow(img)
            plt.title(str(lab))
            plt.show()
        except:
            pass
        
        
        
if __name__ == '__main__':
    read_tfrecords(dataType='train')
