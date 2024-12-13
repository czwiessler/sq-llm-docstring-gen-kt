"""Collection of hyper parameters, network layout, and reporting options."""

import os

import numpy as np
import tensorflow as tf

from asr.labels import num_classes

# Path to git root.
BASE_PATH = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

# Directories:
# Note that the default `train_dir` is outside of the project directory.
tf.flags.DEFINE_string('train_dir',
                       os.path.join(BASE_PATH, '../ctc-asr-checkpoints/3c4r2d-rnn'),
                       "Directory where to write event logs and checkpoints.")

# Note that the default `corpus_dir` is outside of the project directory.
__CORPUS_DIR = os.path.join(BASE_PATH, '../speech-corpus')
tf.flags.DEFINE_string('corpus_dir', os.path.join(__CORPUS_DIR, 'corpus'),
                       "Directory that holds the corpus manifest files.")
tf.flags.DEFINE_string('train_csv', os.path.join(__CORPUS_DIR, 'train.csv'),
                       "Path to the `train.txt` file.")
tf.flags.DEFINE_string('test_csv', os.path.join(__CORPUS_DIR, 'test.csv'),
                       "Path to the `test.txt` file.")
tf.flags.DEFINE_string('dev_csv', os.path.join(__CORPUS_DIR, 'dev.csv'),
                       "Path to the `dev.txt` file.")

# Layer and activation options:
tf.flags.DEFINE_string('used_model', 'ds2',
                       ("Used inference model. Supported are 'ds1', and 'ds2'. "
                        "Also see `FLAGS.feature_drop_every_second_frame`."))

tf.flags.DEFINE_integer('num_units_dense', 2048,
                        "Number of units per dense layer.")
tf.flags.DEFINE_float('relu_cutoff', 20.0,
                      "Cutoff ReLU activations that exceed the cutoff.")

tf.flags.DEFINE_multi_integer('conv_filters', [32, 32, 96],
                              "Number of filters for each convolutional layer.")

tf.flags.DEFINE_integer('num_layers_rnn', 4,
                        "Number of stacked RNN cells.")
tf.flags.DEFINE_integer('num_units_rnn', 2048,
                        "Number of hidden units in each of the RNN cells.")
# TODO: This is currently only implemented for cudnn (`FLAGS.cudnn = True`).
tf.flags.DEFINE_string('rnn_cell', 'rnn_relu',
                       "Used RNN cell type. Supported are the RNN versions 'rnn_relu' and "
                       "'rnn_tanh', as well as the 'lstm' and 'gru' cells")

# Inputs:
tf.flags.DEFINE_integer('batch_size', 16,
                        "Number of samples within a batch.")
tf.flags.DEFINE_string('feature_type', 'mfcc',
                       "Type of input features. Supported types are: 'mel' and 'mfcc'.")
tf.flags.DEFINE_string('feature_normalization', 'local',
                       ("Type of normalization applied to input features."
                        "Supported are: 'none', 'local', and 'local_scalar'"))
tf.flags.DEFINE_boolean('features_drop_every_second_frame', False,
                        "[Deep Speech 1] like dropping of every 2nd input time frame.")

# Learning Rate.
tf.flags.DEFINE_integer('max_epochs', 15,
                        "Number of epochs to run. [Deep Speech 1] uses about 20 epochs.")
tf.flags.DEFINE_float('learning_rate', 1e-5,
                      "Initial learning rate.")
# TODO: The following LR flags are (currently) no longer supported.
tf.flags.DEFINE_float('learning_rate_decay_factor', 4 / 5,
                      "Learning rate decay factor.")
tf.flags.DEFINE_integer('steps_per_decay', 75000,
                        "Number of steps after which learning rate decays.")
tf.flags.DEFINE_float('minimum_lr', 1e-6,
                      "Minimum value the learning rate can decay to.")

# Adam Optimizer:
tf.flags.DEFINE_float('adam_beta1', 0.9,
                      "Adam optimizer beta_1 power.")
tf.flags.DEFINE_float('adam_beta2', 0.999,
                      "Adam optimizer beta_2 power.")
tf.flags.DEFINE_float('adam_epsilon', 1e-8,
                      "Adam optimizer epsilon.")

# CTC decoder:
tf.flags.DEFINE_integer('beam_width', 1024,
                        "Beam width used in the CTC `beam_search_decoder`.")

# Dropout.
tf.flags.DEFINE_float('conv_dropout_rate', 0.0,
                      "Dropout rate for convolutional layers.")
tf.flags.DEFINE_float('rnn_dropout_rate', 0.0,
                      "Dropout rate for the RNN cell layers.")
tf.flags.DEFINE_float('dense_dropout_rate', 0.1,
                      "Dropout rate for dense layers.")

# Corpus:
tf.flags.DEFINE_integer('num_buckets', 96,
                        "The maximum number of buckets to use for bucketing.")

tf.flags.DEFINE_integer('num_classes', num_classes(),
                        "Number of classes. Contains the additional CTC <blank> label.")
tf.flags.DEFINE_integer('sampling_rate', 16000,
                        "The sampling rate of the audio files (e.g. 2 * 8kHz).")

# Performance / GPU:
tf.flags.DEFINE_boolean('cudnn', True,
                        "Whether to use Nvidia cuDNN implementations or the default TensorFlow "
                        "implementation.")
tf.flags.DEFINE_integer('shuffle_buffer_size', 2 ** 14,
                        "Number of elements the dataset shuffle buffer should hold. "
                        "This can consume a large amount of memory.")

# Logging:
tf.flags.DEFINE_integer('log_frequency', 200,
                        "How often (every `log_frequency` steps) to log results.")
tf.flags.DEFINE_integer('num_samples_to_report', 4,
                        "The maximum number of decoded and original text samples to report in "
                        "TensorBoard.")
tf.flags.DEFINE_integer('gpu_hook_query_frequency', 5,
                        "How often (every `gpu_hook_query_frequency` steps) statistics are "
                        "queried from the GPUs.")
tf.flags.DEFINE_integer('gpu_hook_average_queries', 100,
                        "The number of queries to store for calculating average values.")

# Miscellaneous:
tf.flags.DEFINE_boolean('delete', False,
                        "Whether to delete old checkpoints, or resume training.")
tf.flags.DEFINE_integer('random_seed', 0,
                        "TensorFlow random seed. Set to 0 to use the current timestamp instead.")
tf.flags.DEFINE_boolean('log_device_placement', False,
                        "Whether to log device placement.")
tf.flags.DEFINE_boolean('allow_vram_growth', True,
                        "Allow TensorFlow to allocate VRAM as needed, "
                        "as opposed to allocating the whole VRAM at program start.")

# ####### Changing belows parameters is likely to break something. #########
# Export names:
TF_FLOAT = tf.float32  # ctc_* functions don't support float64. See #13
NP_FLOAT = np.float32  # ctc_* functions don't support float64. See #13

# Minimum and maximum length of examples in datasets (in seconds).
MIN_EXAMPLE_LENGTH = 0.7
MAX_EXAMPLE_LENGTH = 17.0

# Feature extraction parameters:
WIN_LENGTH = 0.025  # Window length in seconds.
WIN_STEP = 0.010  # The step between successive windows in seconds.
NUM_FEATURES = 80  # Number of features to extract.

# CSV field names. The field order is always the same as this list from top to bottom.
CSV_HEADER_PATH = 'path'
CSV_HEADER_LABEL = 'label'
CSV_HEADER_LENGTH = 'length'
CSV_FIELDNAMES = [CSV_HEADER_PATH, CSV_HEADER_LABEL, CSV_HEADER_LENGTH]
CSV_DELIMITER = ';'

FLAGS = tf.flags.FLAGS


def get_parameters():
    """
    Generate a summary containing the training, and network parameters.

    Returns:
        str: Summary of training parameters.
    """
    res = '\n\tLearning Rate (lr={}, steps_per_decay={:,d}, decay_factor={});\n' \
          '\tGPU-Options (cudnn={});\n' \
          '\tModel (used_model={}, beam_width={:,d})\n' \
          '\tConv (conv_filters={}); Dense (num_units={:,d});\n' \
          '\tRNN (num_units={:,d}, num_layers={:,d});\n' \
          '\tTraining (batch_size={:,d}, max_epochs={:,d}, log_frequency={:,d});\n' \
          '\tFeatures (type={}, normalization={}, skip_every_2nd_frame={});'

    res = res.format(FLAGS.learning_rate, FLAGS.steps_per_decay, FLAGS.learning_rate_decay_factor,
                     FLAGS.cudnn,
                     FLAGS.used_model, FLAGS.beam_width,
                     FLAGS.conv_filters, FLAGS.num_units_dense,
                     FLAGS.num_units_rnn, FLAGS.num_layers_rnn,
                     FLAGS.batch_size, FLAGS.max_epochs, FLAGS.log_frequency,
                     FLAGS.feature_type, FLAGS.feature_normalization,
                     FLAGS.features_drop_every_second_frame)

    return res
