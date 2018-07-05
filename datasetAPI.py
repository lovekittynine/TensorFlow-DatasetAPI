#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:53:23 2018

@author: wsw
"""

# experiment mnist using tf.data API

import tensorflow as tf
import os
import numpy as np
slim = tf.contrib.slim


tf.reset_default_graph()

dataDir = './mnist'


# 解析一个序列化的样例
def parse_example(serialized_example):
    features = tf.parse_single_example(serialized=serialized_example,
                                       features={'feature':tf.FixedLenFeature([],tf.string),
                                                 'label':tf.FixedLenFeature([],tf.float32)})
    # note dtype must be identical to making tfreocrds tf.uint8
    # 注意解码后的数据类型必须要和制作tfrecords时数据类型一致
    image = tf.decode_raw(features['feature'],tf.uint8)
    image = tf.cast(image,dtype=tf.float32)
    image = image/255.0
    image = tf.reshape(image,shape=[-1,784])
    label = tf.cast(features['label'],dtype=tf.int64)
    # one-hot encoder note:[label]
    label = tf.one_hot([label],depth=10)
    return image,label


# prepare dataset
def DataSet(filename):
    dataset = tf.data.TFRecordDataset(filename)
    return dataset.map(parse_example,num_parallel_calls=4)


def build_model(xs):
    net = slim.fully_connected(xs,num_outputs=512,activation_fn=tf.nn.relu)
    net = slim.fully_connected(net,num_outputs=10,activation_fn=None)
    return net


def train():
    
#    with tf.name_scope('inputs'):
#        xs = tf.placeholder(tf.float32,shape=[None,784])
#        ys = tf.placeholder(tf.int64,shape=[None,])
        
    with tf.name_scope('get_batch'):
        train_dataset = DataSet(os.path.join(dataDir,'train.tfrecords'))
        test_dataset = DataSet(os.path.join(dataDir,'test.tfrecords'))
        # 10 epochs ,batchsize=128
        # 程序以epoch更小的数值为停止条件，所以测试集或者验证集的epoch
        # 不设置（默认无限迭代）
        train_dataset=train_dataset.shuffle(1000).repeat(10).batch(128)
        test_dataset=test_dataset.shuffle(1000).repeat().batch(128)
        # make iterator
        train_iter = train_dataset.make_one_shot_iterator()
        test_iter = test_dataset.make_one_shot_iterator()
        # 要把不同dataset的数据feed进行模型，则需要先创建iterator handle，即iterator placeholder
        handler = tf.placeholder(tf.string,shape=[])
        iterator = tf.data.Iterator.from_string_handle(handler,
                                                       train_dataset.output_types,
                                                       train_dataset.output_shapes)
        image_batch,label_batch = iterator.get_next()
       
    
    with tf.name_scope('model'):
        logits = build_model(image_batch)
        
    with tf.name_scope('losses'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_batch,
                                               logits=logits)
    with tf.name_scope('compute_accuracy'):
        predictions = tf.argmax(logits,axis=-1)
        targets = tf.argmax(label_batch,axis=-1)
        correct_predict = tf.equal(predictions,targets)
        accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32))
        
    with tf.name_scope('optimizer'):
        global_step = tf.train.create_global_step()
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss,global_step)
        
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # generate dataset handle
        # 得到数据类型句柄
        train_handle,test_handle = sess.run([x.string_handle() for x in [train_iter,test_iter]])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            while not coord.should_stop():
                loss_value,train_acc,_ = sess.run([loss,accuracy,train_op],
                                                  feed_dict={handler:train_handle,
                                                             })
                step = global_step.eval()
                fmt = '\r>>>Step:%5d Loss:%5.3f Train_acc:%.3f'%(step,loss_value,train_acc)
                print(fmt,end='',flush=True)
                
                # 每个epoch计算一次测试集ACC
                if step%469==0:
                    # 所有的测试集要遍历一遍，否则只是计算一个batch上的精度(accuracy)
                    test_accu_list = []
                    for i in range(10000//128+1):
                        test_acc = sess.run(accuracy,
                                            feed_dict={handler:test_handle,
                                                       })
                        test_accu_list.append(test_acc)
                    print('\nTest_acc:%.3f' % np.mean(test_accu_list))
                    
        except tf.errors.OutOfRangeError:
              print('\nTraing Finished!!!')
              coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()
