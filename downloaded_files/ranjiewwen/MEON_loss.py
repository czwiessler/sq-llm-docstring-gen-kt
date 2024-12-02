#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created MEON_loss.py by rjw at 19-1-21 in WHU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import gdn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MEON_eval(object):
    def __init__(self, height=256, width=256, channel=3, dist_num=5):
        """
                Args:
                    height: height of image
                    width: width of image
                    channel: number of color channel
                    dist_num: number of distortion types
                    checkpoint_dir: parameter saving directory

                """
        self.height = height
        self.width = width
        self.channel = channel
        self.dist_num = dist_num

    def build_model(self, rgb, reuse=False):

        # params for convolutional layers
        width1 = 5
        height1 = 5
        stride1 = 2
        depth1 = 8

        width2 = 5
        height2 = 5
        stride2 = 2
        depth2 = 16

        width3 = 5
        height3 = 5
        stride3 = 2
        depth3 = 32

        width4 = 3
        height4 = 3
        stride4 = 1
        depth4 = 64

        # params for fully-connected layers
        sub1_fc1 = 128
        sub1_fc2 = self.dist_num

        sub2_fc1 = 256
        sub2_fc2 = self.dist_num

        # convolution layer 1
        with tf.variable_scope('conv1', reuse=reuse):
            weights = tf.get_variable(name='weights',
                                      shape=[height1, width1, self.channel, depth1],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth1],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(rgb, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride1, stride1, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                                    name='pool_x')
            self.feature1 = pool_x  # 64*64

        # convolution layer 2
        with tf.variable_scope('conv2', reuse=reuse):
            weights = tf.get_variable(name='weights',
                                      shape=[height2, width2, depth1, depth2],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth2],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(pool_x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride2, stride2, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                                    name='pool_x')
            self.feature2 = pool_x  # 16*16

        # convolution layer 3
        with tf.variable_scope('conv3', reuse=reuse):
            weights = tf.get_variable(name='weights',
                                      shape=[height3, width3, depth2, depth3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(pool_x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride3, stride3, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                                    name='pool_x')

        # convolution layer 4
        with tf.variable_scope('conv4', reuse=reuse):
            weights = tf.get_variable(name='weights',
                                      shape=[height4, width4, depth3, depth4],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth4],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))

            conv_x = tf.nn.conv2d(input=pool_x, filter=weights, padding='VALID', strides=[1, stride4, stride4, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            conv_out_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                                        name='pool_x')

        # subtask 1
        with tf.variable_scope('subtask1', reuse=reuse):
            with tf.variable_scope('fc1'):
                weights = tf.get_variable(name='weights',
                                          shape=[1, 1, depth4, sub1_fc1],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / math.sqrt(float(depth4))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub1_fc1],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                fc_x = tf.nn.conv2d(input=conv_out_x, filter=weights, padding='VALID', strides=[1, 1, 1, 1],
                                    name='fc_x') + biases
                gdn_x = gdn(inputs=fc_x, inverse=False, data_format='channels_last', name='gdn_x')

            with tf.variable_scope('fc2'):
                weights = tf.get_variable(name='weights',
                                          shape=[1, 1, sub1_fc1, sub1_fc2],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / math.sqrt(float(sub1_fc1))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub1_fc2],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                out_x = tf.squeeze(
                    tf.nn.conv2d(input=gdn_x, filter=weights, padding='VALID', strides=[1, 1, 1, 1]) + biases,
                    name='out_x')

            self.probs = tf.nn.softmax(out_x, name='dist_prob')

        # subtask 2
        with tf.variable_scope('subtask2', reuse=reuse):
            with tf.variable_scope('fc1'):
                weights = tf.get_variable(name='weights',
                                          shape=[1, 1, depth4, sub2_fc1],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / math.sqrt(float(depth4))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub2_fc1],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                fc_x = tf.nn.conv2d(input=conv_out_x, filter=weights, padding='VALID', strides=[1, 1, 1, 1],
                                    name='fc_x') + biases
                gdn_x = gdn(inputs=fc_x, inverse=False, data_format='channels_last', name='gdn_x')

            with tf.variable_scope('fc2'):
                weights = tf.get_variable(name='weights',
                                          shape=[1, 1, sub2_fc1, sub2_fc2],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / math.sqrt(float(sub2_fc1))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub2_fc2],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                self.q_scores = tf.squeeze(tf.nn.conv2d(input=gdn_x, filter=weights, padding='VALID',
                                                        strides=[1, 1, 1, 1]) + biases, name='q_scores')
                self.out_q = tf.reduce_sum(tf.multiply(self.probs, self.q_scores),
                                           axis=1, keep_dims=False, name='out_q')

            # self.saver = tf.train.Saver()
            return self.feature1, self.feature2, self.out_q

    def initialize(self, sess, saver, ckpt_path):

        could_load, checkpoint_counter = self.__load__(sess, saver, ckpt_path)
        if could_load:
            counter = checkpoint_counter
            print('Load successfully!')
        else:
            raise IOError('Fail to load the pretrained model')

        check_init = tf.report_uninitialized_variables()
        assert sess.run(check_init).size == 0

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format('CSIQ', self.dist_num, 'distortions', 'final')  # needs further revision

    def __load__(self, sess, saver, ckpt_dir):
        import re
        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(ckpt_dir, self.model_dir)
        checkpoint_dir = ckpt_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [!] Failed to find a checkpoint")
            return False, 0


def meon_loss(dslr_image, enhanced):
    [w, h, d] = enhanced.get_shape().as_list()[1:]
    MEON_evaluate_model = MEON_eval()
    enhance_f1, enhance_f2, enhance_s = MEON_evaluate_model.build_model(
        255 * tf.image.resize_images(enhanced, [256, 256]), False)
    dslr_f1, dslr_f2, dslr_s = MEON_evaluate_model.build_model(255 * tf.image.resize_images(dslr_image, [256, 256]),
                                                               True)
    loss_content1 = 10000 * tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((enhance_f1 - dslr_f1)))) / (w * h * d))
    loss_content2 = 20000 * tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((enhance_f2 - dslr_f2)))) / (w * h * d))
    loss_content3 = 100 - tf.reduce_mean(enhance_s)
    loss_content = loss_content1 + loss_content2 + loss_content3
    return MEON_evaluate_model, loss_content
