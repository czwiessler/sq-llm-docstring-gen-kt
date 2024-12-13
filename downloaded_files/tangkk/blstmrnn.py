'''
A chord classifier based on bidirectional CTC
'''

from collections import OrderedDict
import os
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from acesongdb import load_data_song

# Set the random number generators' seeds for consistency
SEED = 123

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
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
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


def random_weight(n_in,n_out=0):
    if n_out == 0:
       n_out = n_in 
    W = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
    return W
    
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
    
    return params #[lstm_W, lstm_U, lstm_b]


def lstm_layer(tparams, x, y, use_noise, options, prefix='lstm'):
    n_timesteps = x.shape[0]
    xb = x[::-1]

    def _slice(_x, n, dim):
        return _x[n * dim:(n + 1) * dim]
    
    def _step(x_, xb_, h_, c_, hb_, cb_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += T.dot(x_, tparams[_p(prefix, 'W')])
        preact += tparams[_p(prefix, 'b')]

        i = T.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = T.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        h = o * T.tanh(c)
        
        preactb = T.dot(hb_, tparams[_p(prefix, 'Ub')])
        preactb += T.dot(xb_, tparams[_p(prefix, 'Wb')])
        preactb += tparams[_p(prefix, 'bb')]

        ib = T.nnet.sigmoid(_slice(preactb, 0, options['dim_proj']))
        fb = T.nnet.sigmoid(_slice(preactb, 1, options['dim_proj']))
        ob = T.nnet.sigmoid(_slice(preactb, 2, options['dim_proj']))
        cb = T.tanh(_slice(preactb, 3, options['dim_proj']))

        cb = fb * cb_ + ib * cb
        hb = ob * T.tanh(cb)
        
        # take the reverse of hb and concatenate with h before feeding into logistic regression
        hhb = T.concatenate([h,hb[::-1]])
        # a single frame prediction given h - the posterior probablity
        one_pred = T.nnet.softmax(T.dot(hhb, tparams['U']) + tparams['b'])
        
        return h, c, hb, cb, one_pred
    
    dim_proj = options['dim_proj']
    ydim = options['ydim']
    # the scan function takse one dim_proj vector of x and one target of y at a time
    # the output is:
    # rval[0] -- n_timesteps of h -- n_timesteps * dim_proj
    # rval[1] -- n_timesteps of c -- n_timesteps * dim_proj
    # rval[2] -- n_timesteps of one_pred -- n_timesteps * ydim
    rval, updates = theano.scan(_step,
                                sequences=[x,xb],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           dim_proj),
                                              None],
                                name=_p(prefix, '_layers'),
                                n_steps=n_timesteps)
    
    pred = rval[4]
    pred = T.flatten(pred,2)
    
    trng = RandomStreams(123)
    if options['dropout']:
        pred = dropout_layer(pred, use_noise, trng)
    
    '''
    # CTC forward-backward pass, adapted from:
    # https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
    def recurrence_relation(size):
        big_I = T.eye(size+2)
        return T.eye(size) + big_I[2:,1:-1] + big_I[2:,:-2] * (T.arange(size) % 2)
    
    def path_probs(predict,Y):
        P = predict[:,Y]
        rr = recurrence_relation(Y.shape[0])
        def step(p_curr,p_prev):
            return (p_curr * T.dot(p_prev,rr)).astype(theano.config.floatX)
        probs,_ = theano.scan(
                step,
                sequences = [P],
                outputs_info = [T.eye(Y.shape[0])[0]]
            )
        return probs
        
    def ctc_cost(predict,Y):
        forward_probs  = path_probs(predict,Y)
        backward_probs = path_probs(predict[::-1],Y[::-1])[::-1,::-1]
        probs = forward_probs * backward_probs / predict[:,Y]
        total_prob = T.sum(probs)
        return -T.log(total_prob)
        
    ctc cost - with DP
    cost = ctc_cost(pred, y)
    '''
    
    # pred will be -- n_timesteps * ydim posterior probs
    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
    # pred.argmax(axis=1) will be -- n_timesteps * 1 one hot prediction
    f_pred = theano.function([x], pred.argmax(axis=1), name='f_pred')
    
    # cost will be a scaler value
    # compile a function for the cost for one frame given one_pred and one_y (or y_)
    # cost is a scaler, where y is a n_timesteps * 1 target vector
    off = 1e-8
    cost = -T.log(pred[T.arange(y.shape[0]), y] + off).mean()
    
    return f_pred_prob, f_pred, cost


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, y, cost):
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
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, y, cost):
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

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
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


def rmsprop(lr, tparams, grads, x, y, cost):
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

    f_grad_shared = theano.function([x, y], cost,
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
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.matrix('x', dtype=config.floatX)
    y = T.vector('y', dtype='int64')
    
    f_pred_prob, f_pred, cost = get_layer(options['encoder'])[1](tparams, x, y, use_noise, options,
                                            prefix=options['encoder'])
                                           

    return use_noise, x, y, f_pred_prob, f_pred, cost


def pred_error(f_pred, data, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    # idx0 = numpy.random.randint(0,len(data[0]))
    lendata = len(data[0])
    
    sum_valid_err = 0
    # loop over the valid/test set and predict the error for every song
    for idx0 in range(lendata):
        # on one whole random song
        x = data[0][idx0]
        y = data[1][idx0]

        preds = f_pred(x)
        targets = y
        valid_err = (preds == targets).sum()
        valid_err = 1. - numpy_floatX(valid_err) / len(y)
        sum_valid_err += valid_err
        
    sum_valid_err = sum_valid_err / lendata

    return sum_valid_err

def predprobs(model, X):
    model_options = OrderedDict()
    tparams = init_tparams(model)
    model_options['encoder'] = 'lstm'
    model_options['xdim'] = model['lstm_W'].shape[0]
    model_options['dim_proj'] = model['U'].shape[0]/2
    model_options['ydim'] = model['U'].shape[1]
    model_options['dropout'] = False
    (use_noise, _, _, f_pred_prob, f_pred, _) = build_model(tparams, model_options)
    use_noise.set_value(0.)
    
    return f_pred_prob(X.astype(theano.config.floatX)), f_pred(X.astype(theano.config.floatX))
    
def train_lstm(
    dim_proj=None,
    xdim=None,
    ydim=None,
    patience=10,  # Number of epoch to wait before early stop if no progress
    n_epochs=500,  # The maximum number of epoch to run
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    # n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    trainpath='../data/cv/',
    trainlist='../cvlist/JK-ch-1234-songwise.txt',
    validset='../data/cv/C-ch-songwise.mat',
    dumppath='../model/blstmrnn_model.npz',  # The best model will be saved there
    batch_size=100,  # The batch size during training.

    # Parameter for extra option
    noise_std=0.,
    earlystop=True,
    dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options
    
    print 'Loading data'
    # the dateset is organized as:
    # X - n_songs * n_timesteps * dim_proj (dim_proj = 24 for chromagram based dataset)
    # y - n_songs * n_timesteps * 1
    train, valid = load_data_song(trainpath=trainpath,trainlist=trainlist,validset=validset)
                                   
    print 'data loaded'
    
    model_options['xdim'] = xdim
    model_options['dim_proj'] = dim_proj
    model_options['ydim'] = ydim
    
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
    # the model takes input of:
    # x -- n_timesteps * dim_proj * n_samples (in a simpler case, n_samples = 1 in ctc)
    # y -- n_timesteps * 1 * n_samples (in a simpler case, n_samples = 1 in ctc)
    (use_noise, x,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, y], cost, name='f_cost')

    grads = T.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = T.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, y, cost)

    print 'Optimization'

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])

    best_validation_loss = numpy.inf
    best_p = None    
    # 6000 is a scaling factor assuming every track contains 5000 frames on average
    n_train_batches = len(train[0]) * 5000 / batch_size / 10 # 10 is a scaling factor
    patience = min(10 * n_train_batches,15000)  # look as this many examples regardless
    patience_increase = 1.3    # wait this much longer when a new best is found
    done_looping = False
    improvement_threshold = 0.996  # a relative improvement of this much is  
    validation_frequency = 100 # note here we manually set the validation freq
    training_history = []
    iter = 0
    best_iter = 0
    start_time = time.time()
    for epoch in xrange(n_epochs):
        if earlystop and done_looping:
            print 'early-stopping'
            break
        n_samples = 0

        # Get random sample a piece of length batch_size from a song
        idx0 = numpy.random.randint(0,len(train[0]))
        
        batch_size_ = batch_size
        while len(train[0][idx0]) <= batch_size_:
            batch_size_ = batch_size_ / 2
            
        idx1 = numpy.random.randint(0,len(train[0][idx0])-batch_size_) # 500 in our case
        
        iter += 1
        use_noise.set_value(1.)
        
        # Select the random examples for this minibatch
        x = train[0][idx0][idx1:idx1+batch_size_]
        y = train[1][idx0][idx1:idx1+batch_size_]
        
        # Get the data in numpy.ndarray format
        # This swap the axis!
        # Return something of shape (minibatch maxlen, n samples)
        n_samples += 1

        cost = f_grad_shared(x, y)
        f_update(lrate)


        if numpy.mod(iter, validation_frequency) == 0:
            use_noise.set_value(0.)
            
            this_validation_loss = pred_error(f_pred, valid)
            training_history.append([iter,this_validation_loss])
            print('epoch %i, validation error %f %%' %
                  (epoch, this_validation_loss * 100.))
            print('iter = %d' % iter)
            print('patience = %d' % patience)
            
            if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)
                    
                    params = unzip(tparams)
                    numpy.savez(dumppath, training_history=training_history,
                                best_validation_loss=best_validation_loss,**params)
                        
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    print('best_validation_loss %f' % best_validation_loss)

        if patience <= iter:
                done_looping = True
                if earlystop:
                    break


    end_time = time.time()
    # final save
    numpy.savez(dumppath, training_history=training_history, best_validation_loss=best_validation_loss, **params)

    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
        ) % (best_validation_loss * 100., best_iter + 1)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
    

if __name__ == '__main__':
   
    trainpath = sys.argv[1] #'../data/cv/'
    trainlist = sys.argv[2] #'JK-ch-1234-songwise' hold one out
    validset = sys.argv[3] #'../data/cv/C-ch-songwise.pkl'
    xdim = int(sys.argv[4])#24 or 252
    ydim = int(sys.argv[5])#277
    dim_proj = int(sys.argv[6])#800
    
    # build dumppath
    nntoks = sys.argv[0].split('.')
    nn = nntoks[0]
    dumppath = '../model2/' + nn + '-' + trainlist + '-' + sys.argv[6] + '.npz'
    
    dropout=True
    earlystop=True
    n_epochs = 1000000 # give it long enough time to train
    batch_size = 500 # length of a sample training piece within a song in terms of number of frames
    
    train_lstm(
        dim_proj=dim_proj,
        xdim=xdim,
        ydim=ydim,
        trainpath=trainpath,
        trainlist=trainlist,
        validset=validset,
        dumppath=dumppath,
        n_epochs=n_epochs,
        dropout=dropout,
        earlystop=earlystop,
        batch_size=batch_size
    )
