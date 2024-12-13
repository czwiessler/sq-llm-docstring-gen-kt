# load a .mat file as training, validation, test dataset
# note that scipy.io only deals with .mat file less than 2GB size

import scipy.io as sio
import theano
import theano.tensor as T
import numpy
from sklearn import preprocessing
import h5py

def load_matrix_data_h5py(dataset=None,name=None):
    f = h5py.File(dataset)
    mX = numpy.asarray(f[name]).T
    # this returns X and y equivalent to that of sio's
    return mX

def standardize(train_X, valid_X, test_X, scaling=1, robust=0):
    # FIXME: standardization process can be performed:
    # 1. in terms of column (input variables)
    # 2. in terms of row (input cases)
    # when different features are in different scales, we need to perform 1
    # when different features are in the same scale, but different cases have extranous variations, such as exposure, volumn, dynamics, etc., we need to perform 2.
    # in the usage of ACE data preprocessing, we need to standardize in terms of cases rather than variables
    # see http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html for details
    # for a general workable piece of code, we need to implement both kind of scaling
    ###########################################
    # standardization:
    # for every feature compenent x_i compute the mean(mu) and std(sigma) w.r.t. the whole training X
    # update x_i <= (x_i - mu) / sigma
    # standardize the validation and test set using the same mu and sigma of the training X's
    # mu = numpy.mean(train_X,axis=0)
    # sigma = numpy.std(train_X,axis=0)
    # print "mu %d"%mu.shape
    # print "sigma %d"%sigma.shape
    ## python will automatically broadcast the row vector to the whole matrix, no worry
    # train_X = (train_X - mu) / sigma
    # valid_X = (valid_X - mu) / sigma
    # test_X = (test_X - mu) / sigma
    ##########################################
    
    # scaling = -1: not scaling at all;
    # scaling = 0, perform standardization along axis=0 - scaling input variables
    # scaling = 1, perform standardization along axis=1 - scaling input cases
    if scaling == 0:
        '''
        for every feature compenent x_i compute the mean(mu) and std(sigma) w.r.t. X(:,i)
        update x_i <= (x_i - mu) / sigma
        standardize the validation and test set using the same mu and sigma of the training X's
        both valid and test set need to reuse the scaler of training set
        '''
        if robust == 1:
            scaler = preprocessing.RobustScaler().fit(train_X)
        else:
            scaler = preprocessing.StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        valid_X = scaler.transform(valid_X)
        test_X = scaler.transform(test_X)
    elif scaling == 1:
        '''
        for every training case X compute the mean(mu) and std(sigma) w.r.t. X(i,:)
        update X_i <= (X_i - mu) / sigma
        standardize the validation and test set using the same mu and sigma of the training X's
        training, valid and test set are independently scaled, since scaling is performed in terms of cases
        '''
        if robust == 1:
            train_X = preprocessing.robust_scale(train_X, axis=1)
            valid_X = preprocessing.robust_scale(valid_X, axis=1)
            test_X = preprocessing.robust_scale(test_X, axis=1)
        else:
            train_X = preprocessing.scale(train_X, axis=1)
            valid_X = preprocessing.scale(valid_X, axis=1)
            test_X = preprocessing.scale(test_X, axis=1)
    '''
    if scaling == 1:
        scaler = preprocessing.StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        valid_X = scaler.transform(valid_X)
        test_X = scaler.transform(test_X)
    elif scaling == 2:
    # [0,1] scaling
        min_max_scaler = preprocessing.MinMaxScaler().fit(train_X)
        train_X = min_max_scaler.transform(train_X)
        valid_X = min_max_scaler.transform(valid_X)
        test_X = min_max_scaler.transform(test_X)
    elif scaling == 3:
        # [-1,1] scaling
        max_abs_scaler = preprocessing.MaxAbsScaler().fit(train_X)
        train_X = max_abs_scaler.transform(train_X)
        valid_X = max_abs_scaler.transform(valid_X)
        test_X = max_abs_scaler.transform(test_X)
    '''
    return train_X, valid_X, test_X

def loadmat(trainpath, trainlist, validset, shuffle=0, datasel=1, scaling=1, robust=0, norm=0, h5py=0, share=1):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    # Load the dataset
    # the data is in 'X' 'y' format, where 'X' contains the input features, and 'y' contains the output labels
    
    # concatenate different folds as one according to the trainlist
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
    
    # perform normalization if necessary
    if norm != 0:
        X = preprocessing.normalize(X,norm)

    # transpose the y so that it is a row array instead of a column array, which avoids the error:
    # "cannot convert type tensortype(int32, matrix) into type tensortype(int32,vector)"
    y = y.T[0]
    # put the max of y = 0 (conform with python indexing standard)
    # since originally the data are in .mat format, which conforms with the matlab indexing convention starting from 1
    # rather than 0
    y[y==max(y)]=0
    
    # shuffle the samples
    if shuffle:
        print 'shuffling...........'
        ind = range(0,len(X))
        numpy.random.shuffle(ind)
        X = X[ind]
        y = y[ind]
        
    X = X[0:len(X)*datasel]
    y = y[0:len(y)*datasel]
    
    #############
    # Validation set
    #############
    mat = sio.loadmat(validset)
    Xv = mat['X']
    yv = mat['y']
    yv = yv.T[0]
    yv[yv==max(yv)]=0
    
    
    train_X = numpy.asarray(X,dtype=numpy.float32)
    train_X = preprocessing.scale(train_X, axis=1)
    
    valid_X = numpy.asarray(Xv,dtype=numpy.float32)
    valid_X = preprocessing.scale(valid_X, axis=1)
    
    train_y = numpy.asarray(y,dtype=numpy.int64)
    valid_y = numpy.asarray(yv,dtype=numpy.int64)
    
    # make data set
    train_set = (train_X, train_y)
    valid_set = (valid_X, valid_y)
    
    print "train_X (%d,%d)"%train_X.shape
    print "valid_X (%d,%d)"%valid_X.shape
    
    print "train_y %d"%train_y.shape[0]
    print "valid_y %d"%valid_y.shape[0]
    

    def shared_dataset(data_xy, borrow=True):
        print 'shared_dataset.................'
        
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    if share:
        train_set_x, train_set_y = shared_dataset(train_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
    else:
        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval