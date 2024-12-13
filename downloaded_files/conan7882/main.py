#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: main.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import numpy as np
from scipy import misc
import tensorflow as tf

from utils import load_image
from neural_style import NerualStyle


VGG_PATH = '/home/qge2/workspace/data/pretrain/vgg/vgg19.npy'
STYLE_PATH = '../test_data/'
CONTENT_PATH = '../test_data/'
SAVE_DIR = '../test_data/'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--style', type=str, required=True,
                        help='style image name')
    parser.add_argument('-c', '--content', type=str, required=True,
                        help='content image name')

    parser.add_argument('--cscale', type=int, default=0,
                        help='size of larger side of content image to be rescaled')
    parser.add_argument('--rescale', action='store_true',
                        help='rescale style image to be comparable size of content image')

    parser.add_argument('--wstyle', type=float, default=0.2,
                        help='weight of style cost')
    parser.add_argument('--wcontent', type=float, default=5e-4,
                        help='weight of content cost')
    parser.add_argument('--wvariation', type=float, default=0.1,
                        help='weight of total variation')

    parser.add_argument('--maxiter', type=int, default=500,
                        help='max number of iterations')
    parser.add_argument('--save', action='store_true',
                        help='save result or not')

    return parser.parse_args()


if __name__ == '__main__':

    FLAGS = get_args()
    is_save = FLAGS.save

    # load style and content images
    s_path = os.path.join(STYLE_PATH, FLAGS.style)
    c_path = os.path.join(CONTENT_PATH, FLAGS.content)
    s_im = load_image(s_path, read_channel=3)
    c_im = load_image(c_path, read_channel=3)

    # rescale content image
    if FLAGS.cscale > 0:
        c_shape = list(map(float, c_im.shape[1:3]))
        if c_shape[0] > c_shape[1]:
            c_im = misc.imresize(
                np.squeeze(c_im),
                (int(FLAGS.cscale),
                 int(c_shape[1] * FLAGS.cscale / c_shape[0])))
        else:
            c_im = misc.imresize(
                np.squeeze(c_im),
                (int(c_shape[0] * FLAGS.cscale / c_shape[1]),
                 int(FLAGS.cscale)))
        c_im = np.expand_dims(c_im, axis=0)

    # rescale style image to the size comparable to content image
    if FLAGS.rescale:
        s_shape = list(map(float, s_im.shape[1:3]))
        c_shape = list(map(float, c_im.shape[1:3]))

        r_h = r_w = 0
        if c_shape[0] < s_shape[0]:
            r_h = s_shape[0] / c_shape[0]
        if c_shape[1] < s_shape[1]:
            r_w = s_shape[1] / c_shape[1]
        max_r = max(r_h, r_w)
        if max_r > 0:
            s_im = misc.imresize(np.squeeze(s_im),
                                 (int(s_shape[0] / max_r),
                                  int(s_shape[1] / max_r)))
            s_im = np.expand_dims(s_im, axis=0)

    # init neural style model
    c_h = c_im.shape[1]
    c_w = c_im.shape[2]
    style_trans_model = NerualStyle(pre_train_path=VGG_PATH,
                                    im_height=c_h, im_width=c_w,
                                    content_weight=FLAGS.wcontent,
                                    style_weight=FLAGS.wstyle,
                                    variation_weight=FLAGS.wvariation,
                                    max_iter=FLAGS.maxiter)

    style_trans_model.create_graph()
    if is_save:
        writer = tf.summary.FileWriter(SAVE_DIR)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        initializer = tf.global_variables_initializer()
        sess.run(initializer,
                 feed_dict={style_trans_model.c_im: c_im,
                            style_trans_model.s_im: s_im})

        if is_save:
            writer.add_graph(sess.graph)
        style_trans_model.train_step(sess, is_save, SAVE_DIR)

    if is_save:
        writer.close()
