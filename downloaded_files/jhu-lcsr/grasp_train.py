"""Code for training models on the google brain robotics grasping dataset.

Grasping Dataset:
https://sites.google.com/site/brainrobotdata/home/grasping-dataset

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0


To see help detailing how to run this training script run:

    python grasp_train.py -h

Command line arguments are handled with the [tf flags API](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/platform/flags.py),
which is a simple wrapper around argparse.

"""
import os
import sys
import datetime
import traceback
import numpy as np
import tensorflow as tf
import keras
import json
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.losses import binary_crossentropy

from tensorflow.python.platform import flags

import grasp_dataset
import hypertree_model
import grasp_loss
import hypertree_utilities
import keras_workaround
from callbacks import EvaluateInputTensor
from callbacks import PrintLogsCallback
from callbacks import SlowModelStopping
from callbacks import InaccurateModelStopping

from tqdm import tqdm  # progress bars https://github.com/tqdm/tqdm
from keras_tqdm import TQDMCallback  # Keras tqdm progress bars https://github.com/bstriner/keras-tqdm

try:
    import horovod.keras as hvd
except ImportError:
    print('Horovod is not installed, see https://github.com/uber/horovod.'
          'Distributed training is disabled but single machine training '
          'should continue to work but without learning rate warmup.')
    hvd = None

flags.DEFINE_string('learning_rate_decay_algorithm', 'power_decay',
                    """Determines the algorithm by which learning rate decays,
                       options are power_decay, exp_decay, adam and progressive_drops.
                       Only applies with optimizer flag is SGD""")
flags.DEFINE_string('grasp_model', 'grasp_model_levine_2016_segmentation',
                    """Choose the model definition to run, options are:
                       grasp_model_levine_2016, grasp_model, grasp_model_resnet, grasp_model_segmentation""")
flags.DEFINE_string('save_weights', 'grasp_model_weights',
                    """Save a file with the trained model weights.""")
# flags.DEFINE_string('load_weights', 'grasp_model_weights.h5',
# flags.DEFINE_string('load_weights', 'converted2018-01-20-06-41-24_grasp_model_weights-delta_depth_sin_cos_3-grasp_model_levine_2016-dataset_062_b_063_072_a_082_b_102-epoch-014-val_loss-0.641-val_acc-0.655.h5',
flags.DEFINE_string('load_weights', 'grasp_model_weights.h5',
                    """Load and continue training the specified file containing model weights.""")
flags.DEFINE_integer('epochs', 2,
                     """Epochs of training""")
flags.DEFINE_string('grasp_dataset_test', '097',
                    """Filter the subset of 1TB Grasp datasets to test.
                    097 by default. It is important to ensure that this selection
                    is completely different from the selected training datasets
                    with no overlap, otherwise your results won't be valid!
                    See https://sites.google.com/site/brainrobotdata/home
                    for a full listing.""")
flags.DEFINE_string('grasp_dataset_validation', '092',
                    """Filter the subset of 1TB Grasp datasets for validation.
                    097 by default. It is important to ensure that this selection
                    is completely different from the selected training datasets
                    with no overlap, otherwise your results won't be valid!
                    See https://sites.google.com/site/brainrobotdata/home
                    for a full listing.""")
flags.DEFINE_boolean('test_per_epoch', True,
                     """Do evaluation on dataset_eval above in every epoch.
                        Weight flies for every epoch and single txt file of dataset
                        will be saved.
                     """)
flags.DEFINE_string('pipeline_stage', 'train_eval_test',
                    """Choose to "train", "eval", or "train_eval" with the grasp_dataset
                       data for training and grasp_dataset_test for evaluation.""")
flags.DEFINE_float('learning_rate_scheduler_power_decay_rate', 1.5,
                   """Determines how fast the learning rate drops at each epoch.
                      lr = learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate)
                      Training from scratch within an initial learning rate of 0.1 might find a
                         power decay value of 2 to be useful.
                      Fine tuning with an initial learning rate of 0.001 may consder 1.5 power decay.""")
flags.DEFINE_float('grasp_learning_rate', 0.02,
                   """Determines the initial learning rate""")
flags.DEFINE_float(
    'fine_tuning_learning_rate',
    0.0005,
    'Initial learning rate, this is the learning rate used if load_weights is passed.'
)
flags.DEFINE_integer(
    'fine_tuning_epochs',
    2,
    'Number of epochs to run trainer with all weights marked as trainable.'
)
flags.DEFINE_integer('eval_batch_size', 2, 'batch size per compute device')
flags.DEFINE_integer('densenet_growth_rate', 12,
                     """DenseNet and DenseNetFCN parameter growth rate""")
flags.DEFINE_integer('densenet_depth', 40,
                     """DenseNet total number of layers, aka depth""")
flags.DEFINE_integer('densenet_dense_blocks', 3,
                     """The number of dense blocks in the model.""")
flags.DEFINE_float('densenet_reduction', 0.5,
                   """DenseNet and DenseNetFCN reduction aka compression parameter.""")
flags.DEFINE_float('densenet_reduction_after_pretrained', 0.5,
                   """DenseNet and DenseNetFCN reduction aka compression parameter,
                      applied to the second DenseNet component after pretrained imagenet models.""")
flags.DEFINE_float('dropout_rate', 0.25,
                   """Dropout rate for the model during training.""")
flags.DEFINE_string('eval_results_file', 'grasp_model_eval.txt',
                    """Save a file with results of model evaluation.""")
flags.DEFINE_string('device', '/gpu:0',
                    """Save a file with results of model evaluation.""")
flags.DEFINE_bool('tf_allow_memory_growth', True,
                  """False if memory usage will be allocated all in advance
                     or True if it should grow as needed. Allocating all in
                     advance may reduce fragmentation.""")
flags.DEFINE_string('learning_rate_scheduler', 'learning_rate_scheduler',
                    """Options are None and learning_rate_scheduler,
                       turning this on activates the scheduler which follows
                       a power decay path for the learning rate over time.
                       This is most useful with SGD, currently disabled with Adam.""")
flags.DEFINE_string('optimizer', 'SGD', """Options are Adam and SGD.""")
flags.DEFINE_string('progress_tracker', 'tensorboard',
                    """Utility to follow training progress, options are tensorboard and None.""")
flags.DEFINE_string('loss', 'segmentation_single_pixel_binary_crossentropy',
                    """Options are binary_crossentropy, segmentation_single_pixel_binary_crossentropy,
                       and segmentation_gaussian_binary_crossentropy.""")
flags.DEFINE_string('metric', 'segmentation_single_pixel_binary_accuracy',
                    """Options are accuracy, binary_accuracy and segmentation_single_pixel_binary_accuracy.""")
flags.DEFINE_string('distributed', None,
                    """Options are 'horovod' (github.com/uber/horovod) or None for distributed training utilities.""")
flags.DEFINE_integer('early_stopping', None,
                     """Stop training if the monitored loss does not improve after the specified number of epochs.
                        Values of 0 or None will disable early stopping.
                     """)
flags.DEFINE_string(
    'log_dir',
    './logs_google_brain/',
    'Directory for tensorboard, model layout, model weight, csv, and hyperparam files'
)
flags.DEFINE_string('load_hyperparams', None,
                    """Load hyperparams from a json file. Only applies to grasp_model_hypertree""")
flags.DEFINE_string(
    'run_name',
    '',
    'A string that will become part of the logged directories and filenames.'
)
# flags.FLAGS._parse_flags() not needed for tf 1.5
FLAGS = flags.FLAGS


class GraspTrain(object):

    def __init__(self, tf_session=None, distributed=None):
        """ Create GraspTrain object

            This function configures Keras and the tf session if the tf_session parameter is None.

            # Arguments

            tf_session: The tf session you wish to use, this is reccommended to remain None.
            distributed: The distributed training utility you wish to use, options are 'horovod' and None.
        """
        if distributed is None:
            distributed = FLAGS.distributed
        self.distributed = distributed
        if hvd is not None and self.distributed is 'horovod':
            # Initialize Horovod.
            hvd.init()

        if tf_session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            if hvd is not None and distributed == 'horovod':
                # Pin GPU to be used to process local rank (one GPU per process)
                config.gpu_options.visible_device_list = str(hvd.local_rank())
            # config.inter_op_parallelism_threads = 40
            # config.intra_op_parallelism_threads = 40
            tf_session = tf.Session(config=config)
            K.set_session(tf_session)

    def train(self, dataset=None,
              grasp_datasets_batch_algorithm=None,
              batch_size=None,
              epochs=None,
              test_per_epoch=None,
              load_weights=None,
              save_weights=None,
              make_model_fn=hypertree_model.grasp_model_densenet,
              imagenet_preprocessing=None,
              grasp_sequence_min_time_step=None,
              grasp_sequence_max_time_step=None,
              random_crop=None,
              resize=None,
              resize_height=None,
              resize_width=None,
              learning_rate_decay_algorithm=None,
              learning_rate=None,
              learning_power_decay_rate=None,
              dropout_rate=None,
              model_name=None,
              loss=None,
              metric=None,
              early_stopping=None,
              log_dir=None,
              hyperparams=None,
              run_name=None,
              fine_tuning_learning_rate=None,
              fine_tuning=None,
              fine_tuning_epochs=None,
              pipeline=None,
              test_dataset=None,
              validation_dataset=None,
              checkpoint=True):
        """Train the grasping dataset

        This function depends on https://github.com/fchollet/keras/pull/6928

        # Arguments

            make_model_fn:
                A function of the form below which returns an initialized but not compiled model, which should expect
                input tensors.

                    make_model_fn(pregrasp_op_batch,
                                  grasp_step_op_batch,
                                  simplified_grasp_command_op_batch)

            grasp_sequence_max_time_step: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.
            hyperparams: A dictionary from hyperparameter strings to values. None by default.
        """
        if dataset is None:
            dataset = FLAGS.grasp_datasets_train
        if grasp_datasets_batch_algorithm is None:
            grasp_datasets_batch_algorithm = FLAGS.grasp_datasets_batch_algorithm,
        if batch_size is None:
            batch_size = FLAGS.batch_size
        if epochs is None:
            epochs = FLAGS.epochs
        if test_per_epoch is None:
            test_per_epoch = FLAGS.test_per_epoch
        if load_weights is None:
            load_weights = FLAGS.load_weights
        if save_weights is None:
            save_weights = FLAGS.save_weights
        if make_model_fn is None:
            make_model_fn = hypertree_model.grasp_model_densenet
        if imagenet_preprocessing is None:
            imagenet_preprocessing = FLAGS.imagenet_preprocessing
        if grasp_sequence_min_time_step is None:
            grasp_sequence_min_time_step = FLAGS.grasp_sequence_min_time_step
        if grasp_sequence_max_time_step is None:
            grasp_sequence_max_time_step = FLAGS.grasp_sequence_max_time_step
        if random_crop is None:
            random_crop = FLAGS.random_crop
        if resize is None:
            resize = FLAGS.resize
        if resize_height is None:
            resize_height = FLAGS.resize_height
        if resize_width is None:
            resize_width = FLAGS.resize_width
        if learning_rate_decay_algorithm is None:
            learning_rate_decay_algorithm = FLAGS.learning_rate_decay_algorithm
        if learning_rate is None:
            learning_rate = FLAGS.grasp_learning_rate
        if learning_power_decay_rate is None:
            learning_power_decay_rate = FLAGS.learning_rate_scheduler_power_decay_rate
        if dropout_rate is None:
            dropout_rate = FLAGS.dropout_rate
        if model_name is None:
            model_name = FLAGS.grasp_model
        if loss is None:
            loss = FLAGS.loss
        if metric is None:
            metric = FLAGS.metric
        if early_stopping is None:
            early_stopping = FLAGS.early_stopping
        if test_dataset is None:
            test_dataset = FLAGS.grasp_dataset_test
        if validation_dataset is None:
            validation_dataset = FLAGS.grasp_dataset_validation
        if run_name is None:
            run_name = FLAGS.run_name
        if log_dir is None:
            log_dir = FLAGS.log_dir

        with K.name_scope('train') as scope:
            datasets = dataset.split(',')
            dataset_names_str = dataset.replace(',', '_')
            (pregrasp_op_batch,
             grasp_step_op_batch,
             simplified_grasp_command_op_batch,
             grasp_success_op_batch,
             steps_per_epoch) = grasp_dataset.get_multi_dataset_training_tensors(
                 datasets,
                 batch_size,
                 grasp_datasets_batch_algorithm,
                 imagenet_preprocessing,
                 random_crop,
                 resize,
                 grasp_sequence_min_time_step,
                 grasp_sequence_max_time_step)

            if resize:
                input_image_shape = [batch_size, int(resize_height), int(resize_width), 3]
            else:
                input_image_shape = [batch_size, 512, 640, 3]
            print('input_image_shape: ' + str(input_image_shape))

            run_name = hypertree_utilities.make_model_description(run_name, model_name, hyperparams, dataset_names_str)

            # ###############learning rate scheduler####################
            # source: https://github.com/aurora95/Keras-FCN/blob/master/train.py
            # some quick lines to see what a power_decay schedule would do at each epoch:
            # import numpy as np
            # epochs = 100
            # learning_rate = 0.1
            # learning_power_decay_rate = 2
            # print([learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate) for epoch in np.arange(epochs)])

            def lr_scheduler(epoch, learning_rate=learning_rate,
                             mode=learning_rate_decay_algorithm,
                             epochs=epochs,
                             learning_power_decay_rate=learning_power_decay_rate):
                """if lr_dict.has_key(epoch):
                    lr = lr_dict[epoch]
                    print 'lr: %f' % lr
                """

                if mode is 'power_decay':
                    # original lr scheduler
                    lr = learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate)
                if mode is 'exp_decay':
                    # exponential decay
                    lr = (float(learning_rate) ** float(learning_power_decay_rate)) ** float(epoch+1)
                # adam default lr
                if mode is 'adam':
                    lr = 0.001

                if mode is 'progressive_drops':
                    # drops as progression proceeds, good for sgd
                    if epoch > 0.9 * epochs:
                        lr = 0.0001
                    elif epoch > 0.75 * epochs:
                        lr = 0.001
                    elif epoch > 0.5 * epochs:
                        lr = 0.01
                    else:
                        lr = 0.1

                print('lr: %f' % lr)
                return lr

            loss = self.gather_losses(loss)

            metrics, monitor_metric_name = self.gather_metrics(metric)

            if test_per_epoch:
                monitor_loss_name = 'val_loss'
                monitor_metric_name = 'val_' + monitor_metric_name
            else:
                monitor_loss_name = 'loss'

            callbacks = []
            if hvd is not None and self.distributed is 'horovod':
                callbacks = callbacks + [
                    # Broadcast initial variable states from rank 0 to all other processes.
                    # This is necessary to ensure consistent initialization of all workers when
                    # training is started with random weights or restored from a checkpoint.
                    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

                    # Average metrics among workers at the end of every epoch.
                    #
                    # Note: This callback must be in the list before the ReduceLROnPlateau,
                    # TensorBoard or other metrics-based callbacks.
                    hvd.callbacks.MetricAverageCallback(),
                    # Using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
                    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
                    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
                    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=2, verbose=1)
                ]

            scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)

            # progress_bar = TQDMCallback()
            # callbacks = callbacks + [progress_bar]

            # Will need to try more things later.
            # Nadam parameter choice reference:
            # https://github.com/tensorflow/tensorflow/pull/9175#issuecomment-295395355

            # 2017-08-28 afternoon trying NADAM with higher learning rate
            # optimizer = keras.optimizers.Nadam(lr=0.03, beta_1=0.825, beta_2=0.99685)
            print('FLAGS.optimizer', FLAGS.optimizer)

            # add evalation callback, calls evalation of self.validation_model
            if test_per_epoch:
                print('make_model_fn: ' + str(make_model_fn) + ' model_name: ' + model_name +
                      ' test_per_epoch: ' + str(test_per_epoch) +
                      'validation_dataset: ' + str(validation_dataset) + ' test_dataset: ' + str(test_dataset))
                validation_model, step_num = self.eval(dataset=validation_dataset,
                                                       make_model_fn=make_model_fn,
                                                       model_name=model_name,
                                                       test_per_epoch=test_per_epoch,
                                                       resize_height=resize_height,
                                                       resize_width=resize_width)
                callbacks = callbacks + [EvaluateInputTensor(validation_model, step_num)]
                test_model, step_num = self.eval(dataset=test_dataset,
                                                 make_model_fn=make_model_fn,
                                                 model_name=model_name,
                                                 test_per_epoch=test_per_epoch,
                                                 resize_height=resize_height,
                                                 resize_width=resize_width)
                callbacks = callbacks + [EvaluateInputTensor(test_model, step_num, metrics_prefix='test')]

            if early_stopping is not None and early_stopping > 0.0:
                early_stopper = EarlyStopping(monitor=monitor_loss_name, min_delta=0.001, patience=32)
                callbacks = callbacks + [early_stopper]

            log_dir = os.path.join(log_dir, run_name)
            log_dir_run_name = os.path.join(log_dir, run_name)
            print('Writing logs for models, accuracy and tensorboard in ' + log_dir)
            hypertree_utilities.mkdir_p(log_dir)

            if FLAGS.progress_tracker == 'tensorboard':
                print('Enabling tensorboard in ' + str(log_dir))
                progress_tracker = TensorBoard(log_dir=log_dir, write_graph=True,
                                               write_grads=True, write_images=True)
                callbacks = callbacks + [progress_tracker]

            callbacks += [SlowModelStopping(max_batch_time_seconds=1.5), InaccurateModelStopping()]
            # 2017-08-28 trying SGD
            # 2017-12-18 SGD worked very well and has been the primary training optimizer from 2017-09 to 2018-01
            if FLAGS.optimizer == 'SGD':

                if hvd is not None and self.distributed is 'horovod':
                    # Adjust learning rate based on number of GPUs.
                    multiplier = hvd.size()
                else:
                    multiplier = 1.0

                optimizer = keras.optimizers.SGD(learning_rate * multiplier)
                print(monitor_loss_name)
                callbacks = callbacks + [
                    # Reduce the learning rate if training plateaus.
                    keras.callbacks.ReduceLROnPlateau(patience=4, verbose=1, factor=0.5, monitor=monitor_loss_name)
                ]

            csv_logger = CSVLogger(log_dir_run_name + '.csv')
            callbacks = callbacks + [csv_logger]
            callbacks += [PrintLogsCallback()]

            # Save the hyperparams to a json string so it is human readable
            if hyperparams is not None:
                with open(log_dir_run_name + '_hyperparams.json', 'w') as fp:
                    json.dump(hyperparams, fp)

            if checkpoint:
                checkpoint = keras.callbacks.ModelCheckpoint(
                    log_dir_run_name + '-epoch-{epoch:03d}-' +
                    monitor_loss_name + '-{' + monitor_loss_name + ':.3f}-' +
                    monitor_metric_name + '-{' + monitor_metric_name + ':.3f}.h5',
                    save_best_only=False, verbose=1, monitor=monitor_metric_name)
                callbacks = callbacks + [checkpoint]

            # progress bar
            # callbacks += [TQDMCallback()]
            # 2017-08-27 Tried NADAM for a while with the settings below, only improved for first 2 epochs.
            # optimizer = keras.optimizers.Nadam(lr=0.004, beta_1=0.825, beta_2=0.99685)

            # 2017-12-18, 2018-01-04 Tried ADAM with AMSGrad, great progress initially, but stopped making progress very quickly
            if FLAGS.optimizer == 'Adam':
                optimizer = keras.optimizers.Adam(amsgrad=True)

            if hvd is not None and self.distributed is 'horovod':
                # Add Horovod Distributed Optimizer.
                optimizer = hvd.DistributedOptimizer(optimizer)

            # create the model
            model = make_model_fn(
                pregrasp_op_batch,
                grasp_step_op_batch,
                simplified_grasp_command_op_batch,
                input_image_shape=input_image_shape,
                dropout_rate=dropout_rate)

            # Save the current model to a json string so it is human readable
            with open(log_dir_run_name + '_model.json', 'w') as fp:
                fp.write(model.to_json())

            if load_weights:
                if os.path.isfile(load_weights):
                    model.load_weights(load_weights)
                else:
                    print('Could not load weights {}, '
                          'the file does not exist, '
                          'starting fresh....'.format(load_weights))

            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics,
                          target_tensors=[grasp_success_op_batch])

            print('Available metrics: ' + str(model.metrics_names))

            model.summary()

            # Save the hyperparams to a json string so it is human readable
            if hyperparams is not None:
                with open(log_dir_run_name + '_hyperparams.json', 'w') as fp:
                    json.dump(hyperparams, fp)

            # Save the current model to a json string so it is human readable
            with open(log_dir_run_name + '_model.json', 'w') as fp:
                fp.write(model.to_json())

            try:
                history = model.fit(epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=1)
                final_weights_name = log_dir_run_name + '-final.h5'
                model.save_weights(final_weights_name)
            except (Exception, KeyboardInterrupt) as e:
                ex_type, ex, tb = sys.exc_info()
                traceback.print_tb(tb)
                # deletion must be explicit to prevent leaks
                # https://stackoverflow.com/a/16946886/99379
                del tb
                # always try to save weights
                final_weights_name = log_dir_run_name + '-autosaved-on-exception.h5'
                model.save_weights(final_weights_name)
                raise e
            return final_weights_name, history

    def eval(self, dataset=None,
             batch_size=None,
             load_weights=None,
             save_weights=None,
             make_model_fn=hypertree_model.grasp_model_densenet,
             imagenet_preprocessing=None,
             grasp_sequence_min_time_step=None,
             grasp_sequence_max_time_step=None,
             resize=None,
             resize_height=None,
             resize_width=None,
             eval_results_file=None,
             model_name=None,
             loss=None,
             metric=None,
             test_per_epoch=None):
        """Train the grasping dataset

        This function depends on https://github.com/fchollet/keras/pull/6928

        # Arguments

            make_model_fn:
                A function of the form below which returns an initialized but not compiled model, which should expect
                input tensors.

                    make_model_fn(pregrasp_op_batch,
                                  grasp_step_op_batch,
                                  simplified_grasp_command_op_batch)

            grasp_sequence_max_time_step: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.
            test_per_epoch: A special mode which allows the model to be created for use in a callback which is run every epoch.
                see train() for implementation details.

        # Returns

           weights_name_str or None if a new weights file was not saved.
        """
        with K.name_scope('eval') as scope:
            if dataset is None:
                dataset = FLAGS.grasp_dataset_test
            if batch_size is None:
                batch_size = FLAGS.eval_batch_size
            if load_weights is None:
                load_weights = FLAGS.load_weights
            if save_weights is None:
                save_weights = FLAGS.save_weights
            if make_model_fn is None:
                make_model_fn = hypertree_model.grasp_model_densenet
            if imagenet_preprocessing is None:
                imagenet_preprocessing = FLAGS.imagenet_preprocessing,
            if grasp_sequence_max_time_step is None:
                grasp_sequence_min_time_step = FLAGS.grasp_sequence_min_time_step,
            if grasp_sequence_min_time_step is None:
                grasp_sequence_max_time_step = FLAGS.grasp_sequence_max_time_step,
            if resize is None:
                resize = FLAGS.resize,
            if resize_height is None:
                resize_height = FLAGS.resize_height,
            if resize_width is None:
                resize_width = FLAGS.resize_width,
            if eval_results_file is None:
                eval_results_file = FLAGS.eval_results_file,
            if model_name is None:
                model_name = FLAGS.grasp_model,
            if loss is None:
                loss = FLAGS.loss,
            if metric is None:
                metric = FLAGS.metric,
            if test_per_epoch is None:
                test_per_epoch = FLAGS.test_per_epoch
            if isinstance(dataset, str):
                data = grasp_dataset.GraspDataset(dataset=dataset)
            # TODO(ahundt) ensure eval call to get_training_tensors() always runs in the same order and does not rotate the dataset.
            # list of dictionaries the length of batch_size
            (pregrasp_op_batch, grasp_step_op_batch,
             simplified_grasp_command_op_batch,
             grasp_success_op_batch,
             num_samples) = data.get_training_tensors(batch_size=batch_size,
                                                      imagenet_preprocessing=imagenet_preprocessing,
                                                      random_crop=False,
                                                      image_augmentation=False,
                                                      resize=resize,
                                                      grasp_sequence_min_time_step=grasp_sequence_min_time_step,
                                                      grasp_sequence_max_time_step=grasp_sequence_max_time_step,
                                                      shift_ratio=0.0)

            batch_index_dimension = K.int_shape(grasp_success_op_batch)[0]
            if resize:
                input_image_shape = [batch_index_dimension, resize_height, resize_width, 3]
            else:
                input_image_shape = [batch_index_dimension, 512, 640, 3]

            ########################################################
            # End tensor configuration, begin model configuration and training
            if not test_per_epoch:
                csv_logger = CSVLogger(load_weights + '_eval.csv')

            print('simplified_grasp_command_op_batch: ' + str(simplified_grasp_command_op_batch))
            # create the model
            model = make_model_fn(
                clear_view_image_op=pregrasp_op_batch,
                current_time_image_op=grasp_step_op_batch,
                input_vector_op=simplified_grasp_command_op_batch,
                input_image_shape=input_image_shape,
                dropout_rate=0.0)

            loss = self.gather_losses(loss)

            metrics, monitor_metric_name = self.gather_metrics(metric)

            print('about to compile')
            model.compile(optimizer='sgd',
                          loss=loss,
                          metrics=metrics,
                          target_tensors=[grasp_success_op_batch])
            print('compile complete')

            if not test_per_epoch:
                if(load_weights):
                    if os.path.isfile(load_weights):
                        model.load_weights(load_weights)
                        print('load success')
                    else:
                        raise ValueError('Could not load weights {}, '
                                         'the file does not exist.'.format(load_weights))

            # TODO(ahundt) look back in the history and figure out when/how converted and the return should be used
            # model.save_weights('converted' + load_weights)
            # return

            print('num_samples: ' + str(num_samples) + ' batch size: ' + str(batch_size))
            steps = float(num_samples) / float(batch_size)

            if not steps.is_integer():
                raise ValueError('The number of samples was not exactly divisible by the batch size!'
                                 'For correct, reproducible evaluation your number of samples must be exactly'
                                 'divisible by the batch size. Right now the batch size cannot be changed for'
                                 'the last sample, so in a worst case choose a batch size of 1. Not ideal,'
                                 'but manageable. num_samples: {} batch_size: {}'.format(num_samples, batch_size))

            if test_per_epoch:
                return model, int(steps)

            model.summary()

            try:
                print('evaluating')
                results = model.evaluate(None, None, steps=int(steps))
                # results_str = '\nevaluation results loss: ' + str(loss) + ' accuracy: ' + str(acc) + ' dataset: ' + dataset
                metrics_str = 'metrics:\n' + str(model.metrics_names) + 'results:' + str(results)
                print(metrics_str)
                weights_name_str = load_weights + '_evaluation_dataset_{}_loss_{:.3f}_acc_{:.3f}'.format(dataset, results[0], results[1])
                weights_name_str = weights_name_str.replace('.h5', '') + '.h5'

                results_summary_name_str = weights_name_str.replace('.h5', '') + '.txt'
                with open(results_summary_name_str, 'w') as results_summary:
                    results_summary.write(metrics_str + '\n')
                if save_weights:
                    model.save_weights(weights_name_str)
                    print('\n saved weights with evaluation result to ' + weights_name_str)

            except KeyboardInterrupt as e:
                print('Evaluation canceled at user request, '
                      'any results are incomplete for this run.')
                return None

            return weights_name_str

    def get_compiled_model(self, dataset=None,
                           batch_size=1,
                           load_weights=None,
                           make_model_fn=hypertree_model.grasp_model_densenet,
                           imagenet_preprocessing=None,
                           grasp_sequence_min_time_step=None,
                           grasp_sequence_max_time_step=None,
                           resize=None,
                           resize_height=None,
                           resize_width=None,
                           model_name=None,
                           loss=None,
                           metric=None):
        """ Generate a model and load a dataset and return it for later use.
        """
        with K.name_scope('predict') as scope:
            if dataset is None:
                dataset = FLAGS.grasp_dataset_test
            if load_weights is None:
                load_weights = FLAGS.load_weights
            if make_model_fn is None:
                make_model_fn = hypertree_model.grasp_model_densenet
            if imagenet_preprocessing is None:
                imagenet_preprocessing = FLAGS.imagenet_preprocessing,
            if grasp_sequence_max_time_step is None:
                grasp_sequence_min_time_step = FLAGS.grasp_sequence_min_time_step,
            if grasp_sequence_min_time_step is None:
                grasp_sequence_max_time_step = FLAGS.grasp_sequence_max_time_step,
            if resize is None:
                resize = FLAGS.resize,
            if resize_height is None:
                resize_height = FLAGS.resize_height,
            if resize_width is None:
                resize_width = FLAGS.resize_width,
            if model_name is None:
                model_name = FLAGS.grasp_model,
            if loss is None:
                loss = FLAGS.loss,
            if metric is None:
                metric = FLAGS.metric,
            if isinstance(dataset, str):
                data = grasp_dataset.GraspDataset(dataset=dataset)
            else:
                data = dataset

            (pregrasp_op_batch, grasp_step_op_batch,
             simplified_grasp_command_op_batch,
             grasp_success_op_batch, feature_op_dicts,
             features_complete_list,
             time_ordered_feature_name_dict,
             num_samples) = data.get_training_tensors_and_dictionaries(
                 batch_size=batch_size,
                 imagenet_preprocessing=imagenet_preprocessing,
                 random_crop=False,
                 image_augmentation=False,
                 resize=resize,
                 grasp_sequence_min_time_step=grasp_sequence_min_time_step,
                 grasp_sequence_max_time_step=grasp_sequence_max_time_step,
                 shift_ratio=0.0)

            batch_index_dimension = K.int_shape(grasp_success_op_batch)[0]

            # sometimes we get a tuple of size 1, so extract the number
            if isinstance(resize_height, tuple):
                resize_height = resize_height[0]
            if isinstance(resize_width, tuple):
                resize_width = resize_width[0]

            print('batch_index_dimension ' + str(batch_index_dimension) + ' resize h w ' + str(resize_height) + ' ' + str(resize_width) )
            if resize:
                input_image_shape = [batch_index_dimension, int(resize_height), int(resize_width), 3]
            else:
                input_image_shape = [batch_index_dimension, 512, 640, 3]
            print('input tensor shape pregrasp_op_batch:', K.int_shape(pregrasp_op_batch))
            ########################################################
            # End tensor configuration, begin model configuration and training

            # create the model
            model = make_model_fn(
                clear_view_image_op=pregrasp_op_batch,
                current_time_image_op=grasp_step_op_batch,
                input_vector_op=simplified_grasp_command_op_batch,
                input_image_shape=input_image_shape,
                dropout_rate=0.0)

            loss = self.gather_losses(loss)

            metrics, monitor_metric_name = self.gather_metrics(metric)

            model.compile(optimizer='sgd',
                          loss=loss,
                          metrics=metrics,
                          target_tensors=[grasp_success_op_batch],
                          fetches=feature_op_dicts)

            if(load_weights):
                if os.path.isfile(load_weights):
                    model.load_weights(load_weights, by_name=True, reshape=True)
                else:
                    raise ValueError('Could not load weights {}, '
                                     'the file does not exist.'.format(load_weights))

            # Warning: hacky workaround to get both fetches and predictions back
            # see https://github.com/keras-team/keras/pull/9121 for details
            # TODO(ahundt) remove this hack
            model._make_predict_function = keras_workaround._make_predict_function_get_fetches.__get__(model, Model)
            return (model, pregrasp_op_batch, grasp_step_op_batch,
                    simplified_grasp_command_op_batch,
                    grasp_success_op_batch, feature_op_dicts,
                    features_complete_list,
                    time_ordered_feature_name_dict,
                    num_samples)

    def gather_metrics(self, metric):
        metrics = []
        if 'segmentation_single_pixel_binary_accuracy' in metric:
            monitor_metric_name = metric
            metrics = metrics + [grasp_loss.segmentation_single_pixel_binary_accuracy]
        else:
            metrics = metrics + ['acc']
            monitor_metric_name = 'acc'

        if 'segmentation' in metric:
            metrics = metrics + [grasp_loss.mean_pred_single_pixel]

        metrics = metrics + [grasp_loss.mean_pred, grasp_loss.mean_true]
        return metrics, monitor_metric_name

    def gather_losses(self, loss):
        loss_name = 'loss'
        if isinstance(loss, str) and 'segmentation_single_pixel_binary_crossentropy' in loss:
            loss = grasp_loss.segmentation_single_pixel_binary_crossentropy
            loss_name = 'segmentation_single_pixel_binary_crossentropy'

        elif isinstance(loss, str) and 'segmentation_gaussian_binary_crossentropy' in loss:
            loss = grasp_loss.segmentation_gaussian_binary_crossentropy
            loss_name = 'segmentation_gaussian_binary_crossentropy'
        elif isinstance(loss, (list, tuple)) and len(loss) == 1:
            # strip it to just one element
            [loss] = loss

        return loss


def choose_make_model_fn(grasp_model_name=None, hyperparams=None):
    """ Select the Neural Network Model to use.

        Gets a command line specified function that
        will be used later to create the Keras Model object.

        This function seems a little odd, so please bear with me.
        Instead of generating the model directly, This creates and
        returns a function that will instantiate the model which
        can be called later. In python, functions can actually
        be created and passed around just like any other object.

        Why make a function instead of just creating the model directly now?

        This lets us write custom code that sets up the model
        you asked for in the `--grasp_model` command line argument,
        FLAGS.hypertree_model. This means that when GraspTrain actually
        creates the model they will all work in exactly the same way.
        The end result is GraspTrain doesn't need a bunch of if
        statements for every type of model, and the class can be more focused
        on the grasping datasets and training code.

        # Arguments:

            grasp_model:
                The name of the grasp model to use. Options are
                'grasp_model_resnet'
                'grasp_model_pretrained'
                'grasp_model_densenet'
                'grasp_model_segmentation'
                'grasp_model_levine_2016'

    """
    # defining a temporary variable scope for the callbacks
    class HyperparamCarrier:
        hyperparams = None
    HyperparamCarrier.hyperparams = hyperparams

    if grasp_model_name is None:
        grasp_model_name = FLAGS.grasp_model
    if grasp_model_name == 'grasp_model_resnet':
        def make_model_fn(*a, **kw):
            return hypertree_model.grasp_model_resnet(
                *a, **kw)
    elif grasp_model_name == 'grasp_model_pretrained':
        def make_model_fn(*a, **kw):
            return hypertree_model.grasp_model_pretrained(
                growth_rate=FLAGS.densenet_growth_rate,
                reduction=FLAGS.densenet_reduction_after_pretrained,
                dense_blocks=FLAGS.densenet_dense_blocks,
                *a, **kw)
    elif grasp_model_name == 'grasp_model_densenet':
        def make_model_fn(*a, **kw):
            return hypertree_model.grasp_model_densenet(
                growth_rate=FLAGS.densenet_growth_rate,
                reduction=FLAGS.densenet_reduction,
                dense_blocks=FLAGS.densenet_dense_blocks,
                depth=FLAGS.densenet_depth,
                *a, **kw)
    elif grasp_model_name == 'grasp_model_segmentation':
        def make_model_fn(*a, **kw):
            return hypertree_model.grasp_model_segmentation(
                growth_rate=FLAGS.densenet_growth_rate,
                reduction=FLAGS.densenet_reduction,
                dense_blocks=FLAGS.densenet_dense_blocks,
                *a, **kw)
    elif grasp_model_name == 'grasp_model_levine_2016_segmentation':
        def make_model_fn(*a, **kw):
            return hypertree_model.grasp_model_levine_2016_segmentation(
                *a, **kw)
    elif grasp_model_name == 'grasp_model_levine_2016':
        def make_model_fn(*a, **kw):
            return hypertree_model.grasp_model_levine_2016(
                *a, **kw)
    elif grasp_model_name == 'grasp_model_hypertree':
        def make_model_fn(
                clear_view_image_op=None,
                current_time_image_op=None,
                input_vector_op=None,
                input_image_shape=None,
                **kw):
            print('hyperparams make_model_fn input_image_shape: ' + str(input_image_shape))
            hyperparams = None
            if 'hyperparams' in kw:
                hyperparams = kw['hyperparams']
            elif HyperparamCarrier.hyperparams is None and FLAGS.load_hyperparams:
                hyperparams = hypertree_utilities.load_hyperparams_json(
                    FLAGS.load_hyperparams, FLAGS.fine_tuning, FLAGS.learning_rate)
            elif HyperparamCarrier.hyperparams is not None:
                hyperparams = HyperparamCarrier.hyperparams

            if hyperparams is not None:
                kw.update(hyperparams)
            print('kw: ', kw)
            images = [clear_view_image_op, current_time_image_op]
            vectors = [input_vector_op]
            print('vectors: ' + str(vectors))
            vector_shapes = [K.int_shape(input_vector_op)]
            image_shapes = [input_image_shape] * 2
            print('grasp_model_hypertree image_shapes: ' + str(image_shapes))
            image_model_weights = 'shared'
            print('image_model_weights: ' + image_model_weights)
            # there are hyperparams that are loaded from other areas of the code,
            # so here we remove those that don't apply to the hypertree model directly.
            kw.pop('learning_rate', None)
            kw.pop('batch_size', None)
            kw.pop('feature_combo_name', None)
            # TODO(ahundt) consider making image_model_weights shared vs separate configurable
            return hypertree_model.choose_hypertree_model(
                images=images,
                vectors=vectors,
                image_shapes=image_shapes,
                vector_shapes=vector_shapes,
                image_model_weights=image_model_weights,
                **kw)
    else:
        available_functions = globals()
        if grasp_model_name in available_functions:
            make_model_fn = available_functions[grasp_model_name]
        else:
            raise ValueError('unknown model selected: {}'.format(grasp_model_name))
    return make_model_fn


def run_hyperopt(hyperparams=None, **kwargs):
    """Launch the training and/or evaluation script for the particular model specified on the command line.
    """

    # create the object that does training and evaluation
    # The init() function configures Keras and the tf session if the tf_session parameter is None.
    gt = GraspTrain()

    with K.get_session() as sess:
        model_name = 'grasp_model_hypertree'
        grasp_dataset = '102'
        # Read command line arguments selecting the Keras model to train.
        # The specific Keras Model varies based on the command line arguments.
        # Based on the selection choose_make_model_fn()
        # will create a function that can be called later
        # to actually create a Keras Model object.
        # This is done so GraspTrain doesn't need specific code for every possible Keras Model.
        make_model_fn = choose_make_model_fn(
            grasp_model_name=model_name,
            hyperparams=hyperparams)

        # train the model
        load_weights, history = gt.train(
            make_model_fn=make_model_fn,
            load_weights=None,
            model_name=model_name,
            dataset=grasp_dataset,
            hyperparams=hyperparams)
        return history


def run_training(hyperparams=None):
    """Launch the training and/or evaluation script for the particular model specified on the command line.
    """

    # create the object that does training and evaluation
    # The init() function configures Keras and the tf session if the tf_session parameter is None.
    gt = GraspTrain()

    with K.get_session() as sess:
        # Read command line arguments selecting the Keras model to train.
        # The specific Keras Model varies based on the command line arguments.
        # Based on the selection choose_make_model_fn()
        # will create a function that can be called later
        # to actually create a Keras Model object.
        # This is done so GraspTrain doesn't need specific code for every possible Keras Model.
        make_model_fn = choose_make_model_fn(
            hyperparams=hyperparams)

        # Weights file to load, if any
        load_weights = FLAGS.load_weights

        # train the model
        if 'train' in FLAGS.pipeline_stage:
            print('Training ' + FLAGS.grasp_model)
            load_weights, history = gt.train(make_model_fn=make_model_fn,
                                             load_weights=load_weights,
                                             model_name=FLAGS.grasp_model)
        # evaluate the model
        if 'eval' in FLAGS.pipeline_stage:
            print('Evaluating ' + FLAGS.grasp_model + ' on weights ' + load_weights)
            # evaluate using weights that were just computed, if available
            gt.eval(make_model_fn=make_model_fn,
                    load_weights=load_weights,
                    model_name=FLAGS.grasp_model,
                    test_per_epoch=False)
        return history

if __name__ == '__main__':
    # FLAGS._parse_flags()
    tf.app.run(main=run_training)
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()
