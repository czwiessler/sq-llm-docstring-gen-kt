# author: Kris Zhang
import os
import logging
import warnings
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical, Sequence, multi_gpu_model
from keras import losses
from .modeling import BertConfig, BertModel
from .optimization import AdamWeightDecayOpt
from .checkpoint import StepPreTrainModelCheckpoint


class SampleSequence(Sequence):
    """generator for fitting to pre-training data"""
    def __init__(self, x, y, batch_size, vocab_size, max_predictions_per_seq=20):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_predictions_per_seq = max_predictions_per_seq
        assert len(x) == 4
        assert len(y) == 2

    def __len__(self):
        return int(np.ceil((len(self.x[0]) / self.batch_size)))

    def __getitem__(self, idx):
        batch_x = [
            self.x[0][idx * self.batch_size:(idx + 1) * self.batch_size],
            self.x[1][idx * self.batch_size:(idx + 1) * self.batch_size],
            self.x[2][idx * self.batch_size:(idx + 1) * self.batch_size],
            self.x[3][idx * self.batch_size:(idx + 1) * self.batch_size]
        ]
        batch_y0 = self.y[0][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y0 = batch_y0.reshape((-1))
        batch_y0 = to_categorical(batch_y0, num_classes=self.vocab_size)
        batch_y0 = batch_y0.reshape((-1, self.max_predictions_per_seq, self.vocab_size))
        batch_y = [batch_y0, self.y[1][idx * self.batch_size:(idx + 1) * self.batch_size]]
        return (batch_x, batch_y)


def bert_pretraining(train_data_path, bert_config_file, save_path, batch_size=32, epochs=2, seq_length=128,
                     max_predictions_per_seq=20, lr=5e-5,num_warmup_steps=10000, checkpoints_interval_steps=1000,
                     weight_decay_rate=0.01, validation_ratio=0.1, max_num_val=10000, multi_gpu=0,
                     val_batch_size=None, pretraining_model_name='bert_pretraining.h5',
                     encoder_model_name='bert_encoder.h5', random_state=None):
    '''masked LM/next sentence masked_lm pre-training for BERT.

    # Args
        train_data_path: path of train data.
        bert_config_file: The config json file corresponding to the pre-trained BERT model.
            This specifies the model architecture.
        save_path: dir to save checkpoints.
        batch_size: Integer.  Number of samples per gradient update.
        epochs: Integer. Number of epochs to train the model.
        seq_length: The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, and sequences shorter
            than this will be padded. Must match data generation.
        max_predictions_per_seq:Integer. Maximum number of masked LM predictions per sequence.
        lr: float >= 0. Learning rate.
        num_warmup_steps: Integer. Number of warm up steps.
        checkpoints_interval_steps: Integer. interval of checkpoints. only enable after model is warmed up.
        weight_decay_rate: float. value of weight decay rate.
        validation_ratio:  Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics.
        max_num_val: Integer. max number of validation data.
            when multi_gpu > 0, model will use cpu to evaluate validation data.
            Controlling the argument can benefit the efficiency of validation process.
        multi_gpu: Integer. when multi_gpu > 0, cpu will be use to merge model trained in gpus.
        val_batch_size: Integer.  Number of samples used in validation step.
            If `val_batch_size` is None, val_batch_size will be equal to `batch_size`.
        pretraining_model_name: name of pretraining model file.
        encoder_model_name: name of encoder model file.
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
    '''
    if multi_gpu > 0:
        if not tf.test.is_gpu_available:
            raise ValueError("GPU is not available. Set `multi_gpu` to be 0.")
    pre_training_data = np.load(train_data_path)
    tokens_ids = pre_training_data['tokens_ids']
    tokens_mask = pre_training_data['tokens_mask']
    segment_ids = pre_training_data['segment_ids']
    is_random_next = pre_training_data['is_random_next']
    masked_lm_positions = pre_training_data['masked_lm_positions']
    masked_lm_label = pre_training_data['masked_lm_labels']

    num_train_samples = int(len(tokens_ids) * (1 - validation_ratio))
    num_train_steps = int(np.ceil(num_train_samples / batch_size)) * epochs

    logging.info('train steps: {}'.format(num_train_steps))
    logging.info('train samples: {}'.format(tokens_ids))
    if num_train_steps < num_warmup_steps + checkpoints_interval_steps:
        raise ValueError("number of train steps must be larger than the sum of"
                         " `num_warmup_steps` and `checkpoints_interval_steps`."
                         "enlarge your train data or reduce batch_size")
    warmup_ratio = num_warmup_steps / num_train_steps
    if warmup_ratio > 0.02:
        warnings.warn("model performance may be suitable when warmup steps is 0.01~0.02 of train steps.", UserWarning)

    config = BertConfig.from_json_file(bert_config_file)

    num_val = int(len(tokens_ids) * validation_ratio)
    if num_val > max_num_val:
        validation_ratio = max_num_val / len(tokens_ids)
    # split data for train and valid
    sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_state)
    for train_index, test_index in sss.split(tokens_ids, is_random_next):
        train_tokens_ids, test_tokens_ids = tokens_ids[train_index], tokens_ids[test_index]
        train_tokens_mask, test_tokens_mask = tokens_mask[train_index], tokens_mask[test_index]
        train_segment_ids, test_segment_ids = segment_ids[train_index], segment_ids[test_index]
        train_is_random_next, test_is_random_next = is_random_next[train_index], is_random_next[test_index]
        train_masked_lm_positions, test_masked_lm_positions = masked_lm_positions[train_index], masked_lm_positions[
            test_index]
        train_masked_lm_label, test_masked_lm_label = masked_lm_label[train_index], masked_lm_label[test_index]
        test_masked_lm_label = test_masked_lm_label.reshape((-1))
        test_masked_lm_label = to_categorical(test_masked_lm_label, num_classes=config.vocab_size)
        test_masked_lm_label = test_masked_lm_label.reshape((-1, max_predictions_per_seq, config.vocab_size))

    logging.info("build pretraining nnet...")
    adam = AdamWeightDecayOpt(
        lr=lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        weight_decay_rate=weight_decay_rate,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    )
    bert = BertModel(config,
                     batch_size=batch_size,
                     seq_length=seq_length,
                     max_predictions_per_seq=max_predictions_per_seq,
                     use_token_type=True,
                     embeddings_matrix=None,
                     mask=True
                     )
    if multi_gpu:
        # To avoid OOM errors, this model could have been built on CPU
        with tf.device('/cpu:0'):
            pretraining_model = bert.get_pretraining_model()
            pretraining_model.compile(optimizer=adam, loss=losses.categorical_crossentropy, metrics=['acc'],
                                      loss_weights=[0.5, 0.5])
        parallel_pretraining_model = multi_gpu_model(model=pretraining_model, gpus=multi_gpu)
        parallel_pretraining_model.compile(optimizer=adam, loss=losses.categorical_crossentropy, metrics=['acc'],
                                           loss_weights=[0.5, 0.5])
    else:
        pretraining_model = bert.get_pretraining_model()
        pretraining_model.compile(optimizer=adam, loss=losses.categorical_crossentropy, metrics=['acc'],
                                  loss_weights=[0.5, 0.5])

    logging.info('training pretraining nnet for {} epochs'.format(epochs))
    train_sample_generator = SampleSequence(
        x=[train_tokens_ids, train_tokens_mask, train_segment_ids, train_masked_lm_positions],
        y=[train_masked_lm_label, train_is_random_next],
        batch_size=batch_size,
        vocab_size=config.vocab_size,
        max_predictions_per_seq=max_predictions_per_seq
    )

    checkpoint_model = None
    if multi_gpu:
        checkpoint_model = pretraining_model

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    checkpoint = StepPreTrainModelCheckpoint(
        filepath="%s/%s" % (save_path, pretraining_model_name),
        start_step=num_warmup_steps,
        period=checkpoints_interval_steps,
        save_best_only=True,
        verbose=1,
        val_batch_size=val_batch_size,
        model=checkpoint_model  #when use multi_gpu_model, set model to the original model
    )

    estimator = pretraining_model
    if multi_gpu:
        estimator = parallel_pretraining_model

    estimator.fit_generator(
        generator=train_sample_generator,
        epochs=epochs,
        callbacks=[checkpoint],
        shuffle=False,
        validation_data=([test_tokens_ids, test_tokens_mask, test_segment_ids, test_masked_lm_positions],
                         [test_masked_lm_label, test_is_random_next]),
    )

    pretraining_model.load_weights("%s/%s" % (save_path, pretraining_model_name))
    bert_model = bert.get_bert_encoder()
    bert_model.save_weights("%s/%s" % (save_path, encoder_model_name))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', help='dir of train data')
    parser.add_argument('bert_config_file', help='The config json file corresponding to the pre-trained BERT model')
    parser.add_argument('save_path', help='dir to save checkpoints')
    parser.add_argument('--batch_size', default=32, help='Number of samples per gradient update.')
    parser.add_argument('--epochs', default=2, help='Number of epochs to train the model')
    parser.add_argument('--seq_length', default=128, help='he maximum total input sequence length after tokenization.')
    parser.add_argument('--max_predictions_per_seq', default=20, help='Maximum number of masked LM predictions per sequence.')
    parser.add_argument('--lr', default=5e-5, help='learning rate')
    parser.add_argument('--num_warmup_steps', default=10000, help='number of warm up steps.')
    parser.add_argument('--checkpoints_interval_steps', default=1000, help='interval steps of checkpoints')
    parser.add_argument('--weight_decay_rate', default=0.01, help='value of weight decay rate.')
    parser.add_argument('--validation_ratio', default=0.1, help='Fraction of the training data to be used as validation data')
    parser.add_argument('--max_num_val', default=10000, help='max number of validation data.')
    parser.add_argument('--multi_gpu', default=0, help='number of gpus used to train model.')
    parser.add_argument('--val_batch_size', default=None, help='Number of samples used in validation step.')
    parser.add_argument('--pretraining_model_name', default='bert_pretraining.h5')
    parser.add_argument('--encoder_model_name', default='bert_encoder.h5')
    parser.parse_args()

    bert_pretraining(
        train_data_path=parser.train_data_path,
        bert_config_file=parser.bert_config_file,
        save_path=parser.save_path,
        val_batch_size=parser.val_batch_size
    )



