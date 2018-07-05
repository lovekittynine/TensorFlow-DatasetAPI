#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:33:15 2018

@author: wsw
"""

# make tfrecords

import tensorflow as tf
import numpy as np
import os

tf.reset_default_graph()

trainImages = np.load('./mnist/train.npy')
trainLabels = np.load('./mnist/train-label.npy')

testImages = np.load('./mnist/test.npy')
testLabels = np.load('./mnist/test-label.npy')

def get_tfrecords_example(image,label):
    
    # 一个example的所有特征通过dict构造
    tfrecords_example = {}
    # tf.train.Feature()构建的特征必须使用关键字参数且字段field必须是三种名称之一
    # bytes_list,float_list,int64_list否则会报错
    # tf.train.BytesList(),tf.train.FloatList(),tf.train.Int64List()调用时
    # 也必须使用关键字参数value,且特征值必须放入LIST中
    tfrecords_example['feature'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
    tfrecords_example['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
    # 构建example时也必须使用关键字参数，field=features
    # tf.train.Features()也必须使用关键字参数,field=feature
    example = tf.train.Example(features=tf.train.Features(feature=tfrecords_example))
    serialized_example = example.SerializeToString()
    return serialized_example



def make_tfrecords(dataType='train'):
    
    filename = os.path.join('./mnist','%s.tfrecords'%dataType)
    writer = tf.python_io.TFRecordWriter(path=filename)
    print('Making %s dataset to tfrecords'%dataType)
    
    if dataType=='train':
        total_nums = len(trainLabels)
        for i in range(total_nums):
            image,label = trainImages[i],trainLabels[i]
            serialized_example = get_tfrecords_example(image,label)
            writer.write(serialized_example)
            print('\r>>>>>Num:{:5d}/Total:{:5d}'.format(i+1,total_nums),
                  end='',flush=True)
        writer.close()
     
        
if __name__ == '__main__':
    make_tfrecords(dataType='train')
