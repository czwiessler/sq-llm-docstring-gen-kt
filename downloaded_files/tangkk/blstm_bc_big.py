'''
A chord classifier based on Bidirectional LSTM
'''
# FIXME: please also dump the model trained

from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from acesongdb import load_data_varlen, prepare_data

import balanced_seg

scaling=1
use_dropout=True
max_epochs = 50
batch_size = 100

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    # at training time, use_noise is set to 1,
    # dropout is applied to proj, each unit of proj is presented at a chance of p
    # at test/validation time, use_noise is set to 0,
    # each unit of proj is always presented, and their activations are multiplied by p
    # by default p=0.5 (can be changed)
    # and different p can be applied to different layers, even the input layer
    proj = T.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    '''
    https://en.wikipedia.org/wiki/Word_embedding
    Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing where words from the vocabulary (and possibly phrases thereof) are mapped to vectors of real numbers in a low dimensional space, relative to the vocabulary size ("continuous space").
    '''
    # randn = numpy.random.rand(options['n_words'],
                              # options['dim_proj'])
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    
    # below is for Bidirectional LSTM, the logistic regression needs to accomondate length of 2*dim_proj
    params['U'] = 0.01 * numpy.random.randn(2*options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim1, ndim2):
    W = numpy.random.randn(ndim1, ndim2)
    if ndim1 == ndim2:
        u, s, v = numpy.linalg.svd(W)
        return u.astype(config.floatX)
    elif ndim1 < ndim2:
        u, s, v = numpy.linalg.svd(W,full_matrices=0)
        return v.astype(config.floatX)
    elif ndim1 > ndim2:
        u, s, v = numpy.linalg.svd(W,full_matrices=0)
        return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    # note that this set of 'W','U' and 'b' are for LSTM layer, thus they have prefix in their names
    # W are the input weights
    
    # these are for the forward pass
    W = numpy.concatenate([ortho_weight(options['xdim'],options['dim_proj']),
                           ortho_weight(options['xdim'],options['dim_proj']),
                           ortho_weight(options['xdim'],options['dim_proj']),
                           ortho_weight(options['xdim'],options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W # "lstm_W"
    # U are recurrent weights
    U = numpy.concatenate([ortho_weight(options['dim_proj'],options['dim_proj']),
                           ortho_weight(options['dim_proj'],options['dim_proj']),
                           ortho_weight(options['dim_proj'],options['dim_proj']),
                           ortho_weight(options['dim_proj'],options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U # "lstm_U"
    # b are bias
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX) # "lstm_b"
    
    # these are for the backward pass
    Wb = numpy.concatenate([ortho_weight(options['xdim'],options['dim_proj']),
                           ortho_weight(options['xdim'],options['dim_proj']),
                           ortho_weight(options['xdim'],options['dim_proj']),
                           ortho_weight(options['xdim'],options['dim_proj'])], axis=1)
    params[_p(prefix, 'Wb')] = Wb # "lstm_Wb"
    # U are recurrent weights
    Ub = numpy.concatenate([ortho_weight(options['dim_proj'],options['dim_proj']),
                           ortho_weight(options['dim_proj'],options['dim_proj']),
                           ortho_weight(options['dim_proj'],options['dim_proj']),
                           ortho_weight(options['dim_proj'],options['dim_proj'])], axis=1)
    params[_p(prefix, 'Ub')] = Ub # "lstm_Ub"
    # b are bias
    bb = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'bb')] = bb.astype(config.floatX) # "lstm_bb"
    
    return params #[lstm_W, lstm_U, lstm_b, lstm_Wb, lstm_Ub, lstm_bb]


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    # state_below is word_embedding
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]
    
    # m_ is the mast
    # x_ is passed as the "state_below" (Wx+b)
    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = T.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    # at first, only state_below and mask are available, h and c are not
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

# this is the backward lstm layer (blstm), note that the param set are all ended with 'b'
def blstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    # state_below is word_embedding
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]
    
    # m_ is the mast
    # x_ is passed as the "state_below" (Wx+b)
    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'Ub')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = T.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams[_p(prefix, 'Wb')]) +
                   tparams[_p(prefix, 'bb')])

    dim_proj = options['dim_proj']
    # at first, only state_below and mask are available, h and c are not
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
# lstm_layer is the feedforward layer
# blstm_layer is the backward layer
layers = {'lstm': (param_init_lstm, lstm_layer, blstm_layer)}


def sgd(lr, tparams, grads, x, mask, oh_mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, oh_mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, oh_mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, oh_mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, oh_mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, oh_mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams()

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.matrix('x', dtype=config.floatX)
    mask = T.matrix('mask', dtype=config.floatX)
    oh_mask = T.matrix('oh_mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')
    
    # this part is subject to change for different application domain
    n_timesteps = x.shape[0] / options['xdim'] # x.shape[0] = 400
    n_samples = x.shape[1]
    
    # preparing for constructing backward pass for BLSTM
    bmask = mask[::-1]
    boh_mask = oh_mask[::-1]
    
    '''
    from https://en.wikipedia.org/wiki/Word_embedding
    Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing where words from the vocabulary (and possibly phrases thereof) are mapped to vectors of real numbers in a low dimensional space, relative to the vocabulary size ("continuous space")
    '''
    
    # reshape the feature vector to use LSTM
    emb = x.reshape([n_timesteps, options['xdim'], n_samples])
    emb = numpy.swapaxes(emb,1,2)
    # 2. construct a fliped emb using the fliped input x
    bemb = emb[::-1]
        
    '''
    try to do a bi-directional LSTM using both emb and bemb
    '''
    # forward LSTM pass - with emb and mask
    # get_layer(options['encoder'])[1] gets the calling handle of lstm_layer
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
                                            
    # backward LSTM pass - with bemb and bmask
    # get_layer(options['encoder'])[2] gets the calling handle of blstm_layer
    bproj = get_layer(options['encoder'])[2](tparams, bemb, options,
                                            prefix=options['encoder'],
                                            mask=bmask)
                                            
    # proj is the return of lstm_layer function (rval[0])
    # proj is with the same dimension of emb (it's basically a 1-1 projection of emb?)
    # which is n_timesteps * n_samples * dim_proj
    # we can easily add another layer on top of this proj layer
    if options['encoder'] == 'lstm':
        # this performs a last pooling using oh_mask
        proj = (proj * oh_mask[:, :, None]).sum(axis=0)
        bproj = (bproj * boh_mask[:, :, None]).sum(axis=0)
    
        # this performs a sum pooling using the mask
        # proj = (proj * mask[:, :, None]).sum(axis=0)
        # bproj = (bproj * bmask[:, :, None]).sum(axis=0)
        
        # this performs a mean pooling using the mask
        # proj = (proj * mask[:, :, None]).sum(axis=0)
        # proj = proj / mask.sum(axis=0)[:, None]
        # bproj = (bproj * bmask[:, :, None]).sum(axis=0)
        # bproj = bproj / bmask.sum(axis=0)[:, None]
        
        # this performs a max pooling
        # proj = (proj * mask[:, :, None]).max(axis=0)
        
        # this takes only the last non-zero lstm output (should be natural in modeling chord progression
        # in which the latter frames are influenced by former frames)
        # the last of proj and first of bproj
        # proj = (proj * mask[:, :, None])
        # proj = proj[-1]
        # bproj = (bproj * bmask[:, :, None])
        # bproj = bproj[-1] 
        
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
        bproj = dropout_layer(bproj, use_noise, trng)
    
    # after the above step, both proj and bproj are with dimension n_samples*dim_proj
    
    # this is the logistic regression layer
    # pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])
    # we concatenate both proj and bproj to feed into the logistic regression layer
    # after the concatenation, both cproj is with dimension n_samples*2*dim_proj
    cproj = T.concatenate([proj,bproj],axis=1)
    pred = T.nnet.softmax(T.dot(cproj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask, oh_mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask, oh_mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    
    cost = -T.log(pred[T.arange(y.shape[0]), y] + off).mean()

    return use_noise, x, mask, oh_mask, y, f_pred_prob, f_pred, cost

'''
def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False, maxlen=None):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, oh_mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=maxlen, xdim=xdim)
        pred_probs = f_pred_prob(x, mask, oh_mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs
'''

def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, oh_mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None, xdim=xdim)
        preds = f_pred(x, mask, oh_mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def predprobs(model, X):
    model_options = OrderedDict()
    tparams = init_tparams(model)
    model_options['encoder'] = 'lstm'
    model_options['xdim'] = model['lstm_W'].shape[0]
    model_options['dim_proj'] = model['U'].shape[0]/2
    model_options['ydim'] = model['U'].shape[1]
    model_options['use_dropout'] = False
    (use_noise, _, _, _,
     _, f_pred_prob, f_pred, _) = build_model(tparams, model_options)
     
    use_noise.set_value(0.)
    
    x, mask, oh_mask, _ = prepare_data(X, None, maxlen=None, xdim=model_options['xdim'])
    
    return f_pred_prob(x, mask, oh_mask), f_pred(x, mask, oh_mask)

    
def train_lstm(
    # word embedding in ACE's context can be regarded as the feature vector size of each ns frame
    dim_proj=None,  # word embeding dimension and LSTM number of hidden units.
    xdim=None,
    ydim=None,
    format=None,
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=500,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    # n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    dumppath='blstm_model.npz',  # The best model will be saved there
    validFreq=400,  # Compute the validation error after this number of update.
    saveFreq=1000,  # Save the parameters after every saveFreq updates
    maxlen=None,  # Sequence longer then this get ignored
    batch_size=100,  # The batch size during training.
    valid_batch_size=100,  # The batch size used for validation/test set.
    dataset=None,

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    scaling=1
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    #load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    train, valid, test = load_data_varlen(dataset=dataset, valid_portion=0.1, test_portion=0.1,
                                   maxlen=None, scaling=scaling, robust=0, format=format, h5py=1)
                                   
    print 'data loaded'
    
    '''
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
    '''
    

    ydim = numpy.max(train[1]) + 1
    # ydim = numpy.max(train[1])
    print 'ydim = %d'%ydim

    model_options['ydim'] = ydim
    model_options['xdim'] = xdim
    model_options['dim_proj'] = dim_proj

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, oh_mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)
    
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, oh_mask, y], cost, name='f_cost')

    grads = T.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask, oh_mask, y], grads, name='f_grad')

    lr = T.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, oh_mask, y, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    
    # SP contains an ordered list of (pos), ordered by chord class number [0,ydim-1]
    SP = balanced_seg.balanced_noeval(ydim,train[1])
    
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)
                
                # FIXME: train_index is not used, kf is not used
                bc_idx = balanced_seg.get_bc_idx(SP,ydim)
                # Select the random examples for this minibatch
                y = [train[1][t] for t in bc_idx]
                x = [train[0][t] for t in bc_idx]
                
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, oh_mask, y = prepare_data(x, y, xdim=xdim, maxlen=maxlen)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, oh_mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if dumppath and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',
                    
                    # save the best param set to date (best_p)
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(dumppath, history_errs=history_errs, **params)
                    # pkl.dump(model_options, open('%s.pkl' % dumppath, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    # test_err = pred_error(f_pred, prepare_data, test, kf_test)
                    test_err = 1

                    history_errs.append([valid_err, test_err])
                    
                    # save param only if the validation error is less than the history minimum
                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)
                    
                    # early stopping
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    # test_err = pred_error(f_pred, prepare_data, test, kf_test)
    test_err = 1

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    if dumppath:
        numpy.savez(dumppath, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    dataset = sys.argv[1] #'../data/ch/Jvarlen-ch-noinv.pkl'
    dumppath = sys.argv[2] #'lstm_model.npz'
    format = sys.argv[3] #'matrix'
    xdim = int(sys.argv[4])#24
    dim_proj = int(sys.argv[5])#500
    
    train_lstm(
        dim_proj=dim_proj,
        xdim=xdim,
        format=format,
        dumppath=dumppath,
        max_epochs=max_epochs,
        scaling=scaling,
        dataset=dataset,
        use_dropout=use_dropout,
        batch_size=batch_size,
    )
