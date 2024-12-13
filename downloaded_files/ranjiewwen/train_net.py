#!/usr/bin/env python
# coding: utf-8
#
# Author:   ranjiewen
# URL:
# Created:  2019-01-15

# python train_model.py dataset={iphone,sony,blackberry} dped_dir=dped vgg_dir=pretrain_models/imagenet-vgg-verydeep-19.mat
# model: unet + (L1+SSIM+Muil_Vgg+TV+GAN)

import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy import misc

from data.load_dataset import load_test_data, load_batch
from experiments.config import dped_config_20190305 as config
from loss import color_loss, variation_loss, texture_loss, multi_content_loss
from metrics import MultiScaleSSIM, PSNR
from net import unet
from utils.logger import setup_logger

np.random.seed(0)


def main(args):
    # loading training and test data
    logger.info("Loading test data...")
    test_data, test_answ = load_test_data(args.dataset, args.dataset_dir, args.test_size, args.patch_size)
    logger.info("Test data was loaded\n")

    logger.info("Loading training data...")
    train_data, train_answ = load_batch(args.dataset, args.dataset_dir, args.train_size, args.patch_size)
    logger.info("Training data was loaded\n")

    TEST_SIZE = test_data.shape[0]
    num_test_batches = int(test_data.shape[0] / args.batch_size)

    # defining system architecture
    with tf.Graph().as_default(), tf.Session() as sess:

        # placeholders for training data
        phone_ = tf.placeholder(tf.float32, [None, args.patch_size])
        phone_image = tf.reshape(phone_, [-1, args.patch_height, args.patch_width, 3])

        dslr_ = tf.placeholder(tf.float32, [None, args.patch_size])
        dslr_image = tf.reshape(dslr_, [-1, args.patch_height, args.patch_width, 3])

        adv_ = tf.placeholder(tf.float32, [None, 1])
        enhanced = unet(phone_image)

        # # learning rate exponential_decay
        # global_step = tf.Variable(0)
        # learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps=args.train_size / args.batch_size, decay_rate=0.98, staircase=True)

        # loss introduce
        loss_texture, discim_accuracy = texture_loss(enhanced, dslr_image, args.patch_width, args.patch_height, adv_)
        loss_discrim = -loss_texture
        loss_content = multi_content_loss(args.pretrain_weights, enhanced, dslr_image, args.batch_size)
        loss_color = color_loss(enhanced, dslr_image, args.batch_size)
        loss_tv = variation_loss(enhanced, args.patch_width, args.patch_height, args.batch_size)

        loss_psnr = PSNR(enhanced, dslr_image)
        loss_ssim = MultiScaleSSIM(enhanced, dslr_image)

        loss_generator = args.w_content * loss_content + args.w_texture * loss_texture + args.w_color * loss_color + args.w_tv * loss_tv + 1000*(1-loss_ssim)

        # optimize parameters of image enhancement (generator) and discriminator networks
        generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
        discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]

        train_step_gen = tf.train.AdamOptimizer(args.learning_rate).minimize(loss_generator, var_list=generator_vars)
        train_step_disc = tf.train.AdamOptimizer(args.learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

        saver = tf.train.Saver(var_list=generator_vars, max_to_keep=100)

        logger.info('Initializing variables')
        sess.run(tf.global_variables_initializer())
        logger.info('Training network')
        train_loss_gen = 0.0
        train_acc_discrim = 0.0
        all_zeros = np.reshape(np.zeros((args.batch_size, 1)), [args.batch_size, 1])
        test_crops = test_data[np.random.randint(0, TEST_SIZE, 5), :]

        # summary ,add the scalar you want to see
        tf.summary.scalar('loss_generator', loss_generator),
        tf.summary.scalar('loss_content', loss_content),
        tf.summary.scalar('loss_color', loss_color),
        tf.summary.scalar('loss_texture', loss_texture),
        tf.summary.scalar('loss_tv', loss_tv),
        tf.summary.scalar('discim_accuracy', discim_accuracy),
        tf.summary.scalar('psnr', loss_psnr),
        tf.summary.scalar('ssim', loss_ssim),
        tf.summary.scalar('learning_rate', args.learning_rate),
        merge_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(args.tesorboard_logs_dir, 'train', args.exp_name), sess.graph,
                                             filename_suffix=args.exp_name)
        test_writer = tf.summary.FileWriter(os.path.join(args.tesorboard_logs_dir, 'test', args.exp_name), sess.graph,
                                            filename_suffix=args.exp_name)
        tf.global_variables_initializer().run()

        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        start_i = 0
        if ckpt and ckpt.model_checkpoint_path:
            logger.info('loading checkpoint:' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            import re
            start_i = int(re.findall("_(\d+).ckpt", ckpt.model_checkpoint_path)[0])

        for i in range(start_i, args.iter_max):

            iter_start = time.time()
            # train generator
            idx_train = np.random.randint(0, args.train_size, args.batch_size)
            phone_images = train_data[idx_train]
            dslr_images = train_answ[idx_train]

            [loss_temp, temp] = sess.run([loss_generator, train_step_gen],
                                         feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: all_zeros})
            train_loss_gen += loss_temp / args.eval_step

            # train discriminator
            idx_train = np.random.randint(0, args.train_size, args.batch_size)

            # generate image swaps (dslr or enhanced) for discriminator
            swaps = np.reshape(np.random.randint(0, 2, args.batch_size), [args.batch_size, 1])

            phone_images = train_data[idx_train]
            dslr_images = train_answ[idx_train]
            # sess.run(train_step_disc)=train_step_disc.compute_gradients(loss,var)+train_step_disc.apply_gradients(var) @20190105
            [accuracy_temp, temp] = sess.run([discim_accuracy, train_step_disc],
                                             feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
            train_acc_discrim += accuracy_temp / args.eval_step

            if i % args.summary_step == 0:
                # summary intervals
                train_summary = sess.run(merge_summary,
                                         feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
                train_writer.add_summary(train_summary, i)

            if i % args.eval_step == 0:
                # test generator and discriminator CNNs
                test_losses_gen = np.zeros((1, 7))
                test_accuracy_disc = 0.0

                for j in range(num_test_batches):
                    be = j * args.batch_size
                    en = (j + 1) * args.batch_size

                    swaps = np.reshape(np.random.randint(0, 2, args.batch_size), [args.batch_size, 1])
                    phone_images = test_data[be:en]
                    dslr_images = test_answ[be:en]

                    [enhanced_crops, accuracy_disc, losses] = sess.run([enhanced, discim_accuracy, \
                                                                        [loss_generator, loss_content, loss_color,
                                                                         loss_texture, loss_tv, loss_psnr, loss_ssim]], \
                                                                       feed_dict={phone_: phone_images,
                                                                                  dslr_: dslr_images, adv_: swaps})

                    test_losses_gen += np.asarray(losses) / num_test_batches
                    test_accuracy_disc += accuracy_disc / num_test_batches

                logs_disc = "step %d/%d, %s | discriminator accuracy | train: %.4g, test: %.4g" % \
                            (i, args.iter_max, args.dataset, train_acc_discrim, test_accuracy_disc)
                logs_gen = "generator losses | train: %.4g, test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ssim: %.4g\n" % \
                           (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2],
                            test_losses_gen[0][3], test_losses_gen[0][4], test_losses_gen[0][5], test_losses_gen[0][6])

                logger.info(logs_disc)
                logger.info(logs_gen)

                test_summary = sess.run(merge_summary,
                                        feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
                test_writer.add_summary(test_summary, i)

                # save visual results for several test image crops
                if args.save_visual_result:
                    enhanced_crops = sess.run(enhanced,
                                              feed_dict={phone_: test_crops, dslr_: dslr_images, adv_: all_zeros})
                    idx = 0
                    for crop in enhanced_crops:
                        before_after = np.hstack(
                            (np.reshape(test_crops[idx], [args.patch_height, args.patch_width, 3]), crop))
                        misc.imsave(
                            os.path.join(args.checkpoint_dir, str(args.dataset) + str(idx) + '_iteration_' + str(i) +
                                         '.jpg'), before_after)
                        idx += 1

                # save the model that corresponds to the current iteration
                if args.save_ckpt_file:
                    saver.save(sess,
                               os.path.join(args.checkpoint_dir, str(args.dataset) + '_iteration_' + str(i) + '.ckpt'),
                               write_meta_graph=False)

                train_loss_gen = 0.0
                train_acc_discrim = 0.0
                # reload a different batch of training data
                del train_data
                del train_answ
                del test_data
                del test_answ
                test_data, test_answ = load_test_data(args.dataset, args.dataset_dir, args.test_size, args.patch_size)
                train_data, train_answ = load_batch(args.dataset, args.dataset_dir, args.train_size, args.patch_size)

                # logger.info('current eval_step:{}/{}, take train time is :{}min'.format(i, args.eval_step, float(
                #     time.time() - iter_start) / 60))

            # if KeyboardInterrupt:
            #     saver.save(sess,
            #                os.path.join(args.checkpoint_dir, str(args.dataset) + '_iteration_' + str(i) + '.ckpt'),
            #                write_meta_graph=False)


if __name__ == '__main__':

    args = config.process_command_args(sys.argv[1:])
    # timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
    # args.exp_name=args.exp_name+timestamp
    # args.exp_name="DPED_model_20190101-19:25"

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, str(args.exp_name))
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.isdir(args.tesorboard_logs_dir):
        os.makedirs(args.tesorboard_logs_dir)

    output_dir = args.checkpoint_dir
    logger = setup_logger("TF_EnhanceDPED_benchmark", output_dir)

    logger.info(args)
    start = time.time()
    main(args)
    end = time.time()
    logger.info('total train time is :{}'.format(end - start))
