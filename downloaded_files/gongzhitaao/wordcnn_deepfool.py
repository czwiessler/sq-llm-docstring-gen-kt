import os
import logging
import argparse
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from wordcnn import WordCNN

from utils.core import train, evaluate, predict
from utils.misc import load_data, build_metric
from utils.misc import ReverseEmbedding
from utils.misc import postfn

from attacks import deepfool


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(description='Classify text with WordCNN')

    parser.add_argument('--batch_size', metavar='N', type=int, default=64)
    parser.add_argument('--data', metavar='FILE', type=str, required=True)
    parser.add_argument('--drop_rate', metavar='N', type=float, default=0.2)
    parser.add_argument('--embedding', metavar='FILE', type=str)
    parser.add_argument('--epochs', metavar='N', type=int)
    parser.add_argument('--filters', metavar='N', type=int, default=128)
    parser.add_argument('--indexer', metavar='IDX', type=str)
    parser.add_argument('--kernel_size', metavar='N', type=int, default=3)
    parser.add_argument('--n_classes', metavar='N', type=int, required=True)
    parser.add_argument('--name', metavar='MODEL', type=str)
    parser.add_argument('--outfile', metavar='FILE', type=str, required=True)
    parser.add_argument('--seqlen', metavar='N', type=int, default=300)
    parser.add_argument('--units', metavar='N', type=int, default=512)

    bip = parser.add_mutually_exclusive_group()
    bip.add_argument('--bipolar', dest='bipolar', action='store_true',
                     help='-1/1 for output.')
    bip.add_argument('--unipolar', dest='bipolar', action='store_false',
                     help='0/1 for output.')

    parser.add_argument('--adv_batch_size', metavar='N', type=int, default=64)
    parser.add_argument('--adv_epochs', metavar='N', type=int, default=5)
    parser.add_argument('--adv_eps', metavar='EPS', type=float)
    parser.add_argument('--w2v', metavar='w2v', type=str)

    ka = parser.add_mutually_exclusive_group()
    ka.add_argument('--keepall', dest='keepall', action='store_true',
                     help='save all generated texts.')
    ka.add_argument('--keepadv', dest='keepall', action='store_false',
                     help='save only adversarial texts.')
    parser.set_defaults(keepall=False)

    return parser.parse_args()


def config(args, embedding):
    cfg = deepcopy(args)
    cfg.data = os.path.expanduser(args.data)

    if args.n_classes > 2:
        cfg.output = tf.nn.softmax
    elif 2 == args.n_classes:
        cfg.output = tf.tanh if args.bipolar else tf.sigmoid

    cfg.embedding = tf.placeholder(tf.float32, embedding.shape)
    cfg.vocab_size = embedding.shape[0]
    cfg.embedding_dim = embedding.shape[1]

    cfg.w2v = os.path.expanduser(args.w2v)

    return cfg


def build_graph(cfg):
    class _Dummy:
        pass

    env = _Dummy()

    env.x = tf.placeholder(tf.int32, [None, cfg.seqlen + 1], 'x')
    env.y = tf.placeholder(tf.int32, [None, 1], 'y')
    env.training = tf.placeholder_with_default(False, (), 'mode')

    m = WordCNN(cfg)
    env.ybar = m.predict(env.x, env.training)
    env.model = m

    # we do not save the embedding here since embedding is not trained.
    env.saver = tf.train.Saver(var_list=m.varlist)

    env = build_metric(env, cfg)

    with tf.variable_scope('deepfool'):
        env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
        env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
        env.xadv = deepfool(m, env.x, epochs=env.adv_epochs, eps=env.adv_eps,
                            batch=True, clip_min=-10, clip_max=10)
    return env


def make_adversarial(env, X_data):
    batch_size = env.cfg.adv_batch_size
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    X_sents = []
    for batch in tqdm(range(n_batch), total=n_batch):
        # info('batch {0}/{1}'.format(batch+1, n_batch))
        end = min((batch + 1) * batch_size, n_sample)
        start = end - batch_size
        X_cur = X_data[start:end]
        feed_dict = {env.x: X_cur,
                     env.adv_epochs: env.cfg.adv_epochs,
                     env.adv_eps: env.cfg.adv_eps}
        xadv = env.sess.run(env.xadv, feed_dict=feed_dict)
        inds, sents = env.re.reverse_embedding(xadv, X_cur)
        X_adv[start:end] = inds
        X_sents += sents
    return (X_adv, X_sents)


def main(args):
    info('loading embedding vec')
    embedding = np.load(os.path.expanduser(args.embedding))

    info('constructing config')
    cfg = config(args, embedding)

    info('constructing graph')
    env = build_graph(cfg)
    env.cfg = cfg

    info('initializing session')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(),
             feed_dict={cfg.embedding: embedding})
    sess.run(tf.local_variables_initializer())
    env.sess = sess

    info('loading data')
    (_, _), (X_data, y_data) = load_data(cfg.data, cfg.bipolar, -1)

    info('loading model')
    train(env, load=True, name=cfg.name)
    info('evaluating against clean test samples')
    evaluate(env, X_data, y_data, batch_size=cfg.batch_size)

    env.re = ReverseEmbedding(w2v_file=cfg.w2v, index_file=cfg.indexer)

    info('making adversarial texts')
    X_adv, X_sents = make_adversarial(env, X_data)
    info('evaluating against adversarial texts')
    evaluate(env, X_adv, y_data, batch_size=cfg.batch_size)
    y_adv = predict(env, X_adv, batch_size=cfg.batch_size)
    env.sess.close()
    postfn(cfg, X_sents, y_data, y_adv)


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
