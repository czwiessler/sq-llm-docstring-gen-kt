# this script deals with data that have variable length
# it will dump the processed .mat data to a .pkl for easy access by python

import numpy
import theano
import scipy.io as sio
import theano.tensor as T
from sklearn import preprocessing
from loadmat import standardize
import cPickle
import h5py

def load_matrix_data_h5py(dataset=None,name=None):
    f = h5py.File(dataset)
    mX = numpy.asarray(f[name]).T
    # this returns X and y equivalent to that of sio's
    return mX
    
def load_cell_data_h5py(dataset=None,name=None):
    cX = []
    with h5py.File(dataset) as f:
        for column in f[name]:
            for row_number in range(len(column)):
                dta = numpy.array(f[column[row_number]]).T
                cX.append(dta)
    return cX
    
def load_cell_data_h5py_0(dataset=None,name=None):
    cX = []
    with h5py.File(dataset) as f:
        for column in f[name]:
            row_data = []
            for row_number in range(len(column)):
                dta = f[column[row_number]][:]
                lta = []
                for idx in range(len(dta)):
                    lta.append(dta[idx][0])
                row_data.append(numpy.asarray(lta))
            cX.append(row_data)
    cX = numpy.asarray(cX).T
    return cX

# only this one deals with pure matrix, all other pkl functions are dealing with cell matrix    
def pkl_data_matrix(dataset=None, dumppath=None):
    mat = sio.loadmat(dataset)
    X = mat['X']
    y = mat['y']
    y = y.T[0]
    y[y==max(y)]=0
    
    with open(dumppath, "wb") as f:
        cPickle.dump((X,y), f)
        
    print X.shape
    print y.shape 

# FIXME: y should be in 'uint16' type    
def pkl_data_varlen_h5py(dataset=None, dumppath=None, fdim=None, nseg=1):
    X = load_cell_data_h5py_0(dataset,'X')
    X = X.T
    cX = []
    for i in range(len(X)):
        xi = X[i][0]
        # reshape xi and then average every nseg in n_step
        xr = xi.reshape(len(xi)/fdim, fdim) # n_step * n_feature
        xm = []
        for j in range(xr.shape[0])[0::nseg]: # start from index 0 and 
            xtemp = xr[j:numpy.min([j+nseg,xr.shape[0]])]
            xm.append(numpy.mean(xtemp,axis=0))
        xm = numpy.asarray(xm)
        xm = xm.flatten()
        # always standardize every cases when using cell as input
        # since this seems to be a reasonable thing to do
        xm = preprocessing.scale(xm)
        cX.append(xm)
    X = numpy.asarray(cX)
    
    y = load_matrix_data_h5py(dataset,'y')
    y = y.T[0]
    y[y==max(y)]=0
    
    print X.shape
    print y.shape
    
    with open(dumppath, "wb") as f:
        cPickle.dump((X,y), f)
        
def pkl_data_varlen(dataset=None, dumppath=None, fdim=None, nseg=1):
    mat = sio.loadmat(dataset)
    X = mat['X']
    cX = []
    for i in range(len(X)):
        xi = X[i][0][0]
        # reshape xi and then average every nseg in n_step
        xr = xi.reshape(len(xi)/fdim, fdim) # n_step * n_feature
        xm = []
        for j in range(xr.shape[0])[0::nseg]: # start from index 0 and 
            xtemp = xr[j:numpy.min([j+nseg,xr.shape[0]])]
            xm.append(numpy.mean(xtemp,axis=0))
        xm = numpy.asarray(xm)
        xm = xm.flatten()
        # always standardize every cases when using cell as input
        # since this seems to be a reasonable thing to do
        xm = preprocessing.scale(xm)
        cX.append(xm)
    X = numpy.asarray(cX)
    
    y = mat['y']
    y = y.T[0]
    y[y==max(y)]=0
    
    print X.shape
    print y.shape
    
    with open(dumppath, "wb") as f:
        cPickle.dump((X,y), f)
    
def pkl_data_nseg_h5py(dataset=None, dumppath=None, nseg=6, ymax=None):
    
    #############
    # LOAD DATA #
    #############
    X = load_cell_data_h5py(dataset,'X') # X is a list instead of a numpy array
    y = load_cell_data_h5py(dataset,'y')
    
    y_ = []
    for k in range(len(y)): # for every song entry
        # y_k is the target for one song "12 * n_timesteps" for all 12 keys transposition
        y_k = y[k]
        for kk in range(12): # iterate over 12 keys
            # y_kk is one song targets for one key
            ym = y_k[kk]
            # target no need to standardize
            # assuming max value of target is ymax
            ym[ym==ymax]=0
            ym = ym.astype('uint16')
            y_.append(ym)
            # y_ stores targets for all songs all keys, the 12 key dim is unrolled
    y = numpy.asarray(y_)
    
    print y.shape
    print y[0].shape
    
    # X dimension is 1 * number of songs
    X_ = []
    for k in range(len(X)): # for every song entry   
        # X_k is the feature for one song "12 * n_timesteps * dim_proj" for all 12 keys transposition
        X_k = X[k]
        for kk in range(12): # iterate over 12 keys
            X__ = []
            # X_kk is the feature for one key "n_timesteps * dim_proj" for the song
            X_kk = X_k[kk]
            lenkk = X_kk.shape[0]
            for i in range(X_kk.shape[0]):
                # xm is one time step of X_k. The size of xi is always dim_proj
                xmm = X_kk[i:numpy.min([i+nseg,lenkk])]
                xm = numpy.mean(xmm,axis=0)
                xm = preprocessing.scale(xm).astype(theano.config.floatX)
                # X__ stores features for one song one key
                X__.append(xm)
            # X_ stores features for all songs all keys, the 12 key dim is unrolled
            X_.append(numpy.asarray(X__))
    X = numpy.asarray(X_)
    
    print X.shape
    print X[0].shape
    print X[0][0].shape
    
    with open(dumppath, "wb") as f:
        cPickle.dump((X,y), f)
        
def pkl_data_waveform_h5py(dataset=None, dumppath=None, ymax=None):

    #############
    # LOAD DATA #
    #############
    X = load_cell_data_h5py(dataset,'X') # X is a list instead of a numpy array
    y = load_cell_data_h5py(dataset,'y')

    y_ = []
    for k in range(len(y)): # for every song entry
        # y_k is the target for one song "12 * n_timesteps" for all 12 keys transposition
        y_k = y[k]
        for kk in range(12): # iterate over 12 keys
            # y_kk is one song targets for one key
            ym = y_k[kk]
            # target no need to standardize
            ym[ym==ymax]=0
            ym = ym.astype('uint16')
            y_.append(ym)
            # y_ stores targets for all songs all keys, the 12 key dim is unrolled
    y = numpy.asarray(y_)
    
    print y.shape
    print y[0].shape
    
    # similar to y
    X_ = []
    for k in range(len(X)): # for every song entry
        # y_k is the target for one song "12 * n_timesteps" for all 12 keys transposition
        X_k = X[k]
        for kk in range(12): # iterate over 12 keys
            # y_kk is one song targets for one key
            Xm = X_k[kk]
            # target no need to standardize
            Xm = Xm.astype(theano.config.floatX)
            X_.append(Xm)
            # y_ stores targets for all songs all keys, the 12 key dim is unrolled
    X = numpy.asarray(X_)
    
    print X.shape
    print X[0].shape
    
    with open(dumppath, "wb") as f:
        cPickle.dump((X,y), f)
        
def pkl_data_framewise_h5py(dataset=None, dumppath=None, ymax=None):

    #############
    # LOAD DATA #
    #############
    X = load_cell_data_h5py(dataset,'X') # X is a list instead of a numpy array
    y = load_cell_data_h5py(dataset,'y')

    y_ = []
    for k in range(len(y)): # for every song entry
        # y_k is the target for one song "12 * n_timesteps" for all 12 keys transposition
        y_k = y[k]
        for kk in range(12): # iterate over 12 keys
            # y_kk is one song targets for one key
            ym = y_k[kk]
            # target no need to standardize
            ym[ym==ymax]=0
            ym = ym.astype('uint16')
            y_.append(ym)
            # y_ stores targets for all songs all keys, the 12 key dim is unrolled
    y = numpy.asarray(y_)
    
    print y.shape
    print y[0].shape
    
    # X dimension is 1 * number of songs
    X_ = []
    for k in range(len(X)): # for every song entry
        # X_k is the feature for one song "12 * n_timesteps * dim_proj" for all 12 keys transposition
        X_k = X[k]
        for kk in range(12): # iterate over 12 keys
            X__ = []
            # X_kk is the feature for one key "n_timesteps * dim_proj" for the song
            X_kk = X_k[kk]
            for i in range(X_kk.shape[0]):
                # xm is one time step of X_k. The size of xi is always dim_proj
                xm = X_kk[i]
                xm = preprocessing.scale(xm).astype(theano.config.floatX)
                # X__ stores features for one song one key
                X__.append(xm)
            # X_ stores features for all songs all keys, the 12 key dim is unrolled
            X_.append(numpy.asarray(X__))
    X = numpy.asarray(X_)
    
    print X.shape
    print X[0].shape
    print X[0][0].shape
    
    with open(dumppath, "wb") as f:
        cPickle.dump((X,y), f)
        
def pkl_data_nseg(dataset=None, dumppath=None, nseg=6, ymax=None):
    
    #############
    # LOAD DATA #
    #############
    mat = sio.loadmat(dataset)
    X = mat['X']
    
    # X dimension is 1 * number of songs
    X_ = []
    for k in range(X.shape[1]): # for every song entry   
        # X_k is the feature for one song "12 * n_timesteps * dim_proj" for all 12 keys transposition
        X_k = X[0][k]
        for kk in range(12): # iterate over 12 keys
            X__ = []
            # X_kk is the feature for one key "n_timesteps * dim_proj" for the song
            X_kk = X_k[kk]
            lenkk = X_kk.shape[0]
            for i in range(X_kk.shape[0]):
                # xm is one time step of X_k. The size of xi is always dim_proj
                xmm = X_kk[i:numpy.min([i+nseg,lenkk])]
                xm = numpy.mean(xmm,axis=0)
                xm = preprocessing.scale(xm).astype(theano.config.floatX)
                # X__ stores features for one song one key
                X__.append(xm)
            # X_ stores features for all songs all keys, the 12 key dim is unrolled
            X_.append(numpy.asarray(X__))
    X = numpy.asarray(X_)
    
    print X.shape
    print X[0].shape
    print X[0][0].shape
    
    y = mat['y']
    y_ = []
    for k in range(y.shape[1]): # for every song entry
        # y_k is the target for one song "12 * n_timesteps" for all 12 keys transposition
        y_k = y[0][k]
        for kk in range(12): # iterate over 12 keys
            # y_kk is one song targets for one key
            ym = y_k[kk]
            # target no need to standardize
            # assuming max value of target is ymax
            ym[ym==ymax]=0
            y_.append(ym)
            # y_ stores targets for all songs all keys, the 12 key dim is unrolled
    y = numpy.asarray(y_)
    
    print y.shape
    print y[0].shape
    
    with open(dumppath, "wb") as f:
        cPickle.dump((X,y), f)

def pkl_data_framewise(dataset=None, dumppath=None, ymax=None):

    #############
    # LOAD DATA #
    #############
    mat = sio.loadmat(dataset)
    X = mat['X']
    
    # X dimension is 1 * number of songs
    X_ = []
    for k in range(X.shape[1]): # for every song entry   
        # X_k is the feature for one song "12 * n_timesteps * dim_proj" for all 12 keys transposition
        X_k = X[0][k]
        for kk in range(12): # iterate over 12 keys
            X__ = []
            # X_kk is the feature for one key "n_timesteps * dim_proj" for the song
            X_kk = X_k[kk]
            for i in range(X_kk.shape[0]):
                # xm is one time step of X_k. The size of xi is always dim_proj
                xm = X_kk[i]
                xm = preprocessing.scale(xm).astype(theano.config.floatX)
                # X__ stores features for one song one key
                X__.append(xm)
            # X_ stores features for all songs all keys, the 12 key dim is unrolled
            X_.append(numpy.asarray(X__))
    X = numpy.asarray(X_)
    
    print X.shape
    print X[0].shape
    print X[0][0].shape
    
    y = mat['y']
    y_ = []
    for k in range(y.shape[1]): # for every song entry
        # y_k is the target for one song "12 * n_timesteps" for all 12 keys transposition
        y_k = y[0][k]
        for kk in range(12): # iterate over 12 keys
            # y_kk is one song targets for one key
            ym = y_k[kk]
            # target no need to standardize
            # assuming max value of target is ymax
            ym[ym==ymax]=0
            y_.append(ym)
            # y_ stores targets for all songs all keys, the 12 key dim is unrolled
    y = numpy.asarray(y_)
    
    print y.shape
    print y[0].shape
    
    with open(dumppath, "wb") as f:
        cPickle.dump((X,y), f)

def prepare_data(seqs, labels, maxlen=None, xdim=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)
    # print 'maxlen is %d'%maxlen

    x = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    # the length of each submask should be maxlen/xdim (assuming x is a 1-dim vector where maxlen already contains
    # the xdim dimension, which means maxlen is dividable by xdim)
    x_mask = numpy.zeros((maxlen/xdim, n_samples)).astype(theano.config.floatX)
    x_oh_mask = numpy.zeros((maxlen/xdim, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs): # this enumerate n_samples
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx]/xdim, idx] = 1. # full hot mask
        # x_oh_mask[lengths[idx]/xdim-1, idx] = 1. # one hot mask
        
    x_oh_mask = x_mask
    return x, x_mask, x_oh_mask, labels

def load_data_varlen(trainpath, trainlist, validset,
                     valid_portion=0.1, test_portion=0.1, maxlen=None,
                     scaling=1, robust=0, format='matrix', h5py=0):
    '''Loads the dataset

    :type dataset: String
    :param dataset: The path to the dataset (here ACE dataset)
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############
   
    if format == 'cell':
        # FIXME: do not support 'cell' yet.
        f = open(dataset, 'rb')
        X,y = cPickle.load(f)
    elif format == 'matrix':
        rc = 0
        if h5py == 0:
            strtoks = trainlist.split('-')
            sets = strtoks[0]
            folds = strtoks[1]
            feature = strtoks[2]
            print strtoks
            if len(strtoks) > 3:
                nseg = strtoks[3]
                print nseg
            for fold in folds:
                if len(strtoks) == 3:
                    dataset = trainpath + sets + '-' + fold + '-' + feature
                else:
                    dataset = trainpath + sets + '-' + fold + '-' + feature + '-' + nseg
                print dataset
                if rc == 0:
                    mat = sio.loadmat(dataset)
                    X = mat['X']
                    y = mat['y']
                    rc = rc + 1
                else:
                    mat = sio.loadmat(dataset)
                    X = numpy.concatenate((X,mat['X']))
                    y = numpy.concatenate((y,mat['y']))
        else:
            strtoks = trainlist.split('-')
            sets = strtoks[0]
            folds = strtoks[1]
            feature = strtoks[2]
            print strtoks
            if len(strtoks) > 3:
                nseg = strtoks[3]
                print nseg
            for fold in folds:
                if len(strtoks) == 3:
                    dataset = trainpath + sets + '-' + fold + '-' + feature
                else:
                    dataset = trainpath + sets + '-' + fold + '-' + feature + '-' + nseg
                print dataset
                if rc == 0:
                    X = load_matrix_data_h5py(dataset,'X')
                    y = load_matrix_data_h5py(dataset,'y')
                    rc = rc + 1
                else:
                    X = numpy.concatenate((X,load_matrix_data_h5py(dataset,'X')))
                    y = numpy.concatenate((y,load_matrix_data_h5py(dataset,'y')))
        y = y.T[0]
        y[y==max(y)]=0
          
    if format == 'matrix':
        X = preprocessing.scale(X, axis=1)
    train = (X,y)
    
    
    mat = sio.loadmat(validset)
    Xv = mat['X']
    yv = mat['y']
    yv = yv.T[0]
    yv[yv==max(yv)]=0
    if format == 'matrix':
        Xv = preprocessing.scale(Xv, axis=1)
    valid = (Xv,yv)


    return train, valid
    
def load_data_song(trainpath, trainlist, validset):
    strtoks = trainlist.split('-')
    sets = strtoks[0]
    folds = strtoks[1]
    feature = strtoks[2]
    songwise = strtoks[3]
    print strtoks
    rc = 0
    for fold in folds:
        dataset = trainpath + sets + '-' + fold + '-' + feature + '-' + songwise + '.pkl'
        print dataset
        if rc == 0:
            f = open(dataset, 'rb')
            X,y = cPickle.load(f)
            rc = rc + 1
            f.close()
        else:
            f = open(dataset, 'rb')
            X_,y_ = cPickle.load(f)
            X = numpy.concatenate((X,X_))
            y = numpy.concatenate((y,y_))
            f.close()
    
    # do not shuffle
    train = (X,y)

    fv = open(validset, 'rb')
    Xv,yv = cPickle.load(fv)
    # take only every 12 entries
    valid = (Xv[0::12],yv[0::12])
    fv.close()

    return train, valid
    

if __name__ == '__main__':
    load_data_h5py()