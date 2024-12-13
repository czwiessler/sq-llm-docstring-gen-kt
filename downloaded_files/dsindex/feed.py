from __future__ import print_function
import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np

def build_feed_dict(model, dataset, max_sentence_length, is_train):
    """Build feed_dict for dataset
    """ 
    config = model.config
    feed_dict={model.input_data_pos_ids: dataset['pos_ids'],
               model.input_data_chk_ids: dataset['chk_ids'],
               model.output_data: dataset['tags'],
               model.is_train: is_train,
               model.sentence_length: max_sentence_length}
    feed_dict[model.input_data_word_ids] = dataset['word_ids']
    feed_dict[model.input_data_wordchr_ids] = dataset['wordchr_ids']
    if 'elmo' in config.emb_class:
        feed_dict[model.elmo_input_data_wordchr_ids] = dataset['elmo_wordchr_ids']
    if 'bert' in config.emb_class:
        feed_dict[model.bert_input_data_token_ids] = dataset['bert_token_ids']
        feed_dict[model.bert_input_data_token_masks] = dataset['bert_token_masks']
        feed_dict[model.bert_input_data_segment_ids] = dataset['bert_segment_ids']
    return feed_dict

def build_input_feed_dict(model, bucket, Input):
    """Build input and feed_dict for bucket(inference only), by default, with model
    """
    config = model.config
    inp = Input(bucket, config, build_output=False)
    feed_dict = {model.input_data_pos_ids: inp.example['pos_ids'],
                 model.input_data_chk_ids: inp.example['chk_ids'],
                 model.is_train: False,
                 model.sentence_length: inp.max_sentence_length}
    feed_dict[model.input_data_word_ids] = inp.example['word_ids']
    feed_dict[model.input_data_wordchr_ids] = inp.example['wordchr_ids']
    if 'elmo' in config.emb_class:
        feed_dict[model.elmo_input_data_wordchr_ids] = inp.example['elmo_wordchr_ids']
    if 'bert' in config.emb_class:
        feed_dict[model.bert_input_data_token_ids] = inp.example['bert_token_ids']
        feed_dict[model.bert_input_data_token_masks] = inp.example['bert_token_masks']
        feed_dict[model.bert_input_data_segment_ids] = inp.example['bert_segment_ids']
    return inp, feed_dict

def build_input_feed_dict_with_graph(graph, config, bucket, Input):
    """Build input and feed_dict for bucket(inference only) with graph
    """
    # mapping placeholders
    p_is_train = graph.get_tensor_by_name('prefix/is_train:0')
    p_sentence_length = graph.get_tensor_by_name('prefix/sentence_length:0')
    p_input_data_pos_ids = graph.get_tensor_by_name('prefix/input_data_pos_ids:0')
    p_input_data_chk_ids = graph.get_tensor_by_name('prefix/input_data_chk_ids:0')
    p_input_data_word_ids = graph.get_tensor_by_name('prefix/input_data_word_ids:0')
    p_input_data_wordchr_ids = graph.get_tensor_by_name('prefix/input_data_wordchr_ids:0')
    if 'elmo' in config.emb_class:
        p_elmo_input_data_wordchr_ids = graph.get_tensor_by_name('prefix/elmo_input_data_wordchr_ids:0')
    if 'bert' in config.emb_class:
        p_bert_input_data_token_ids = graph.get_tensor_by_name('prefix/bert_input_data_token_ids:0')
        p_bert_input_data_token_masks = graph.get_tensor_by_name('prefix/bert_input_data_token_masks:0')
        p_bert_input_data_segment_ids = graph.get_tensor_by_name('prefix/bert_input_data_segment_ids:0')

    inp = Input(bucket, config, build_output=False)
    feed_dict = {p_input_data_pos_ids: inp.example['pos_ids'],
                 p_input_data_chk_ids: inp.example['chk_ids'],
                 p_is_train: False,
                 p_sentence_length: inp.max_sentence_length}
    feed_dict[p_input_data_word_ids] = inp.example['word_ids']
    feed_dict[p_input_data_wordchr_ids] = inp.example['wordchr_ids']
    if 'elmo' in config.emb_class:
        feed_dict[p_elmo_input_data_wordchr_ids] = inp.example['elmo_wordchr_ids']
    if 'bert' in config.emb_class:
        feed_dict[p_bert_input_data_token_ids] = inp.example['bert_token_ids']
        feed_dict[p_bert_input_data_token_masks] = inp.example['bert_token_masks']
        feed_dict[p_bert_input_data_segment_ids] = inp.example['bert_segment_ids']
    return inp, feed_dict

def align_bert_embeddings(config, bert_embeddings, bert_wordidx2tokenidx, idx):
    """Align bert_embeddings via bert_wordidx2tokenidx
         ex) word  : 'johanson was a guy to'          [0 ~ 4]
             token : 'johan ##son was a gu ##y t ##o' [0 ~ 7]
             wordidx2tokenidx : [1 3 4 5 7 9 0 0 ...] (bert embedding begins with [CLS] token)
             bert embedding :   [em('CLS'), em('johan'), em('##son'), em('was'), em('a'), em('gu'), em('##y'), em('t'), em('##o'), 0, ...]
    """
    def mean_pooling(ls):
        '''Reduce by averaging along with rows.
             Args:
               ls: list of embedding
           code from https://github.com/Adaxry/get_aligned_BERT_emb/blob/master/get_aligned_bert_emb.py#L27
        '''
        if len(ls) == 1:
            return ls[0]
        for item in ls[1:]:
            for index, value in enumerate(item):
                ls[0][index] += value
        return [value / len(ls) for value in ls[0]]

    def mean_pooling_with_cls(ls, cls):
        '''Reduce by averaging along with rows.
             Args:
               ls: list of embedding
               cls: '[CLS]' sentence embedding for BERT
        '''
        for item in ls:
            for index, value in enumerate(item):
                cls[index] += value
        return [value / (len(ls)+1) for value in cls]

    if idx == 0:
        tf.logging.debug('# bert_embeddings')
        t = bert_embeddings[0]
        tf.logging.debug(' '.join([str(x) for x in np.shape(t)]))
        t = bert_embeddings[0][0][1] # first (batch, seq, token) embedding
        tf.logging.debug(' '.join([str(x) for x in t]))

    # 4-dim -> 3-dim
    bert_embeddings = bert_embeddings[0]    

    bert_embeddings_updated = []
    batch_size = len(bert_wordidx2tokenidx)
    for i in range(batch_size): # batch
        bert_embedding_updated = []
        prev = 1
        for j in range(len(bert_wordidx2tokenidx[i])): # seq
            cur = bert_wordidx2tokenidx[i][j]
            if j == 0:
                prev = cur
                continue        # skip first for '[CLS]'
            if cur == 0: break  # process before padding area
            
            # mean prev ~ cur
            try:
                pooled = mean_pooling(bert_embeddings[i][prev:cur])
                '''
                cls = bert_embeddings[i][0]
                pooled = mean_pooling_with_cls(bert_embeddings[i][prev:cur], cls)
                '''
                bert_embedding_updated.append(pooled)
            except:
                tf.logging.debug('[ERROR] ' + 'seq:' + str(i) + '\t' + 'prev:' + str(prev) + '\t' + 'cur:' + str(cur))
                # error padding
                padding = [0.0] * config.bert_dim
                bert_embedding_updated.append(padding)
            
            prev = cur
        # padding
        while len(bert_embedding_updated) < config.bert_max_seq_length:
            padding = [0.0] * config.bert_dim
            bert_embedding_updated.append(padding)
        bert_embeddings_updated.append(bert_embedding_updated)

    if idx == 0:
        tf.logging.debug('# bert_embeddings_updated')
        t = bert_embeddings_updated[0][0] # first (batch, seq, token) embedding
        tf.logging.debug(' '.join([str(x) for x in t]))
        tf.logging.debug('# batch size: ' + str(len(bert_embeddings_updated)))
        tf.logging.debug('# seq size: ' + str(len(bert_embeddings_updated[0])))
        tf.logging.debug('# emb size: ' + str(len(bert_embeddings_updated[0][0])))

    return bert_embeddings_updated

