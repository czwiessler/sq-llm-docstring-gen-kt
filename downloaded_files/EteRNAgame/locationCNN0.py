# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:49:27 2017

@author: rohankoodli
Creates, trains, and saves the CNN for predicting location of nucleotide change
Trains on 92 key puzzles from the 99th percentile of Eterna players (Players who've solved > 3000 puzzles)
"""

import numpy as np
import os
import tensorflow as tf
import pickle
from getData import getPid

NAME = 'locationCNN28'
NUM_FEATURES = 8
TRAIN_KEEP_PROB = 0.9
TEST_KEEP_PROB = 1.0
learning_rate = 0.0001
ne = 150
tb_path = '/tensorboard/' + NAME

train = 30000
test = 100
abs_max = 350
len_puzzle = abs_max

TF_SHAPE = NUM_FEATURES * len_puzzle

# enc0 = np.array([[[[1,2,3,4],[0,1,0,1],[-33,0,0,0]],[[1,2,3,4],[0,1,1,0],[-23,0,0,0]]],[[[3,3,3,3],[0,0,0,0],[2,0,0,0]],[[1,1,1,0],[1,0,1,0],[-23,0,0,0]]]])
# ms0 = np.array([[[2,1],[4,3]],[[1,6],[2,9]]])
# enc = np.array([[[1,2,3,4],[0,1,0,1],[1,1,1,1],[-3,0,0,0]],[[4,3,2,1],[1,0,1,0],[0,0,0,0],[9,0,0,0]]])
# out = np.array([[4,2],[3,3]])

with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
content = [int(x) for x in content]
progression = [6502966,6502968,6502973,6502976,6502984,6502985,6502993, \
                6502994,6502995,6502996,6502997,6502998,6502999,6503000] # 6502957
content.extend(progression)
#content = getPid()
content.remove(6502966)
content.remove(6502976)
content.remove(6502984)
# content.remove(4960718)
# content.remove(3468526)
# content.remove(3468547)
#content.remove(3522605)

real_X = []
real_y = []
pids = []

for pid in content:
    try:
        feats = pickle.load(open(os.getcwd()+'/pickles/X5-exp-loc-'+str(pid),'rb'))
        yloc = pickle.load(open(os.getcwd()+'/pickles/y5-exp-loc-'+str(pid),'rb'))
        real_X.extend(feats)
        real_y.extend(yloc)
        pids.append(feats)
    except IOError:
        continue

print "Unpickled"

# real_X = features6502997 + features6502995 #+ features6502990 + features6502996 + features6502963 + features6502964 \
#          #+ features6502966 + features6502967 + features6502968 + features6502969 + features6502970 + features6502976
# real_y = labels6502997 + labels6502995 #+ labels6502990 + labels6502996 + labels6502963 + labels6502964 \
#          #+ labels6502966 + labels6502967 + labels6502968 + labels6502969 + labels6502970 + labels6502976
# max_lens = []
# pids = [features6502997,features6502995]#,features6502990,features6502996,features6502963,features6502964, \
#         #features6502966,features6502967,features6502968,features6502969,features6502970,features6502976]

max_lens = []
for puzzle in pids:
    max_lens.append(len(puzzle[0][0]))

abs_max = 350
indxs = []
for i in range(len(max_lens)):
     if max_lens[i] < abs_max: #max(max_lens):
         indxs.append(i)

for i in indxs:
     if pids[i]:
         for j in pids[i]:
             for k in j:
                 k.extend([0]*(abs_max - len(k))) #k.extend([0]*(max(max_lens) - len(k)))

# for i in real_y:
#     i.extend([0]*50)

print abs_max

'''
Used for testing accuracies without certain features
'''
# for i in real_X:
#     del i[5]
#     del i[5]

print len(real_X), len(real_y)
print np.array(real_X).shape, np.array(real_y).shape

#testtest = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])

'''
Uncomment for altering training data (training on half experts)
'''

cx = ([real_X[i * 36:(i + 1) * 36] for i in range((len(real_X) + 36 - 1) // 36 )])
cy = ([real_y[i * 36:(i + 1) * 36] for i in range((len(real_y) + 36 - 1) // 36 )])
cx.pop()
cy.pop()
print np.array(cx).shape

print np.array(cx).shape
print np.array(cy).shape
real_X_9 = np.array(cx[:587]).reshape((-1,TF_SHAPE))
real_y_9 = np.array(cy[:587]).reshape(587, 350)
test_real_X = np.array(cx[587:30420]).reshape((-1,TF_SHAPE))
test_real_y = np.array(cy[587:30420]).reshape(29833, 350)
print len(real_X_9)
print real_X_9.shape
print real_y_9.shape
print test_real_X.shape
print test_real_y.shape

real_X_9 = np.array(real_X[0:train]).reshape([-1,TF_SHAPE])
real_y_9 = np.array(real_y[0:train])
test_real_X = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])
test_real_y = np.array(real_y[train:train+test])

print "Data prepped"

# real_X_9, test_real_X, real_y_9, test_real_y = np.array(train_test_split(real_X[0:train],real_y[0:train],test_size=0.01))
# real_X_9, test_real_X, real_y_9, test_real_y = np.array(real_X_9).reshape([-1,TF_SHAPE]), np.array(test_real_X).reshape([-1,TF_SHAPE]), np.array(real_y_9), np.array(test_real_y)

# enc0 = np.array([[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]]])
# ms0 = np.array([[1,6],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7]])
# ms0 = np.array([[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) # just base
#
# test_enc0 = np.array([[[2,3,3,2],[0,0,0,0],[6,0,0,0],[0,0,1,1]],[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]]])
# test_ms0 = np.array([[4,20],[3,15]])
# test_ms0 = np.array([[0,0,0,1],[1,0,0,0]]) # just base

n_classes = 350
batch_size = 100 # load 100 features at a time

x = tf.placeholder('float',[None,TF_SHAPE],name="x_placeholder") # 216 with enc0
y = tf.placeholder('float',name='y_placeholder')
keep_prob = tf.placeholder('float',name='keep_prob_placeholder')

# enc = enc0.reshape([-1,16])
# ms = ms0#.reshape([-1,4])
#
# test_enc = test_enc0.reshape([-1,16])
# test_ms = test_ms0

#e1 = tf.reshape(enc0,[])


def conv2d(x,W):
    """
    2D convolutions helper function
    :param x: Inputs
    :param W: Weights
    :return: Tensorflow convolution
    """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def maxpool2d(x):
    """
    2D pooling helper function
    :param x: Inputs
    :return: Tensorflow pooling
    """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


def convNeuralNet(x):
    """
    Setting up the CNN architecture
    :param x: Features placeholder
    :return: convolutions and fully-connected
    """
    weights = {'w_conv1':tf.get_variable('w_conv1',[NUM_FEATURES,NUM_FEATURES,1,2],initializer=tf.random_normal_initializer()),
               'w_conv2':tf.get_variable('w_conv2',[NUM_FEATURES,NUM_FEATURES,2,4],initializer=tf.random_normal_initializer()),
               'w_conv3':tf.get_variable('w_conv3',[NUM_FEATURES,NUM_FEATURES,4,8],initializer=tf.random_normal_initializer()),
               'w_conv4':tf.get_variable('w_conv4',[NUM_FEATURES,NUM_FEATURES,8,16],initializer=tf.random_normal_initializer()),
               'w_conv5':tf.get_variable('w_conv5',[NUM_FEATURES,NUM_FEATURES,16,32],initializer=tf.random_normal_initializer()),
               'w_conv6':tf.get_variable('w_conv6',[NUM_FEATURES,NUM_FEATURES,32,64],initializer=tf.random_normal_initializer()),
               'w_conv7':tf.get_variable('w_conv7',[NUM_FEATURES,NUM_FEATURES,64,128],initializer=tf.random_normal_initializer()),
               'w_conv8':tf.get_variable('w_conv8',[NUM_FEATURES,NUM_FEATURES,128,256],initializer=tf.random_normal_initializer()),
               'w_conv9':tf.get_variable('w_conv9',[NUM_FEATURES,NUM_FEATURES,256,512],initializer=tf.random_normal_initializer()),
               'w_conv10':tf.get_variable('w_conv10',[NUM_FEATURES,NUM_FEATURES,512,1024],initializer=tf.random_normal_initializer()),
            #    'w_conv11':tf.get_variable('w_conv11',[8,8,1024,2048],initializer=tf.random_normal_initializer()),
            #    'w_conv12':tf.get_variable('w_conv12',[8,8,2048,4096],initializer=tf.random_normal_initializer()),
            #    'w_conv13':tf.get_variable('w_conv13',[8,8,4096,8192],initializer=tf.random_normal_initializer()),
            #    'w_conv14':tf.get_variable('w_conv14',[8,8,8192,16384],initializer=tf.random_normal_initializer()),
            #    'w_conv15':tf.get_variable('w_conv15',[8,8,16384,32768],initializer=tf.random_normal_initializer()),
               'w_fc1':tf.get_variable('w_fc1',[1024,1024],initializer=tf.random_normal_initializer()),
               'w_fc2':tf.get_variable('w_fc2',[1024,2048],initializer=tf.random_normal_initializer()),
               'w_fc3':tf.get_variable('w_fc3',[2048,4096],initializer=tf.random_normal_initializer()),
               'w_fc4':tf.get_variable('w_fc4',[4096,8192],initializer=tf.random_normal_initializer()),
               # 'w_fc5':tf.get_variable('w_fc5',[8192,16384],initializer=tf.random_normal_initializer()),
               'out':tf.get_variable('w_out',[8192,n_classes],initializer=tf.random_normal_initializer())}

    biases = {'b_conv1':tf.get_variable('b_conv1',[2],initializer=tf.random_normal_initializer()),
              'b_conv2':tf.get_variable('b_conv2',[4],initializer=tf.random_normal_initializer()),
              'b_conv3':tf.get_variable('b_conv3',[8],initializer=tf.random_normal_initializer()),
              'b_conv4':tf.get_variable('b_conv4',[16],initializer=tf.random_normal_initializer()),
              'b_conv5':tf.get_variable('b_conv5',[32],initializer=tf.random_normal_initializer()),
              'b_conv6':tf.get_variable('b_conv6',[64],initializer=tf.random_normal_initializer()),
              'b_conv7':tf.get_variable('b_conv7',[128],initializer=tf.random_normal_initializer()),
              'b_conv8':tf.get_variable('b_conv8',[256],initializer=tf.random_normal_initializer()),
              'b_conv9':tf.get_variable('b_conv9',[512],initializer=tf.random_normal_initializer()),
              'b_conv10':tf.get_variable('b_conv10',[1024],initializer=tf.random_normal_initializer()),
            #   'b_conv11':tf.get_variable('b_conv11',[2048],initializer=tf.random_normal_initializer()),
            #   'b_conv12':tf.get_variable('b_conv12',[4096],initializer=tf.random_normal_initializer()),
            #   'b_conv13':tf.get_variable('b_conv13',[8192],initializer=tf.random_normal_initializer()),
            #   'b_conv14':tf.get_variable('b_conv14',[16384],initializer=tf.random_normal_initializer()),
            #   'b_conv15':tf.get_variable('b_conv15',[32768],initializer=tf.random_normal_initializer()),
              'b_fc1':tf.get_variable('b_fc1',[1024],initializer=tf.random_normal_initializer()),
              'b_fc2':tf.get_variable('b_fc2',[2048],initializer=tf.random_normal_initializer()),
              'b_fc3':tf.get_variable('b_fc3',[4096],initializer=tf.random_normal_initializer()),
              'b_fc4':tf.get_variable('b_fc4',[8192],initializer=tf.random_normal_initializer()),
              # 'b_fc5':tf.get_variable('b_fc5',[16384],initializer=tf.random_normal_initializer()),
              'out':tf.get_variable('b_out',[n_classes],initializer=tf.random_normal_initializer())}

    x = tf.reshape(x,shape=[-1,NUM_FEATURES,len_puzzle,1])

    conv1 = tf.nn.sigmoid(conv2d(x, weights['w_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.sigmoid(conv2d(conv1, weights['w_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.sigmoid(conv2d(conv2, weights['w_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)

    conv4 = tf.nn.sigmoid(conv2d(conv3, weights['w_conv4']) + biases['b_conv4'])
    conv4 = maxpool2d(conv4)

    conv5 = tf.nn.sigmoid(conv2d(conv4, weights['w_conv5']) + biases['b_conv5'])
    conv5 = maxpool2d(conv5)

    conv6 = tf.nn.sigmoid(conv2d(conv5, weights['w_conv6']) + biases['b_conv6'])
    conv6 = maxpool2d(conv6)

    conv7 = tf.nn.sigmoid(conv2d(conv6, weights['w_conv7']) + biases['b_conv7'])
    conv7 = maxpool2d(conv7)

    conv8 = tf.nn.sigmoid(conv2d(conv7, weights['w_conv8']) + biases['b_conv8'])
    conv8 = maxpool2d(conv8)

    conv9 = tf.nn.sigmoid(conv2d(conv8, weights['w_conv9']) + biases['b_conv9'])
    conv9 = maxpool2d(conv9)

    conv10 = tf.nn.sigmoid(conv2d(conv9, weights['w_conv10']) + biases['b_conv10'])
    conv10 = maxpool2d(conv10)

    # conv11 = conv2d(conv10, weights['w_conv11'])
    # conv11 = maxpool2d(conv11)
    #
    # conv12 = conv2d(conv11, weights['w_conv12'])
    # conv12 = maxpool2d(conv12)
    #
    # conv13 = conv2d(conv12, weights['w_conv13'])
    # conv13 = maxpool2d(conv13)
    #
    # conv14 = conv2d(conv13, weights['w_conv14'])
    # conv14 = maxpool2d(conv14)
    #
    # conv15 = conv2d(conv14, weights['w_conv15'])
    # conv15 = maxpool2d(conv15)

    fc1 = tf.reshape(conv10, [-1, 1024])
    fc1 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, weights['w_fc1']), biases['b_fc1']))

    fc2 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, weights['w_fc2']), biases['b_fc2']))

    fc3 = tf.nn.sigmoid(tf.add(tf.matmul(fc2, weights['w_fc3']), biases['b_fc3']))

    fc4 = tf.nn.sigmoid(tf.add(tf.matmul(fc3, weights['w_fc4']), biases['b_fc4']))

    #fc5 = tf.nn.sigmoid(tf.add(tf.matmul(fc4,weights['w_fc5']),biases['b_fc5']))

    last = tf.nn.dropout(fc4,keep_prob)

    #output = tf.add(tf.matmul(fc,weights['out']),biases['out'],name='final')
    output = tf.add(tf.matmul(last, weights['out']), biases['out'], name='op7')

    return output

    tf.summary.histogram('weights1',weights['w_conv1'])
    tf.summary.histogram('biases1',biases['b_conv1'])
    tf.summary.histogram('act1',conv1)

    tf.summary.histogram('weights2',weights['w_conv2'])
    tf.summary.histogram('biases2',biases['b_conv2'])
    tf.summary.histogram('act2',conv2)

    tf.summary.histogram('weights3',weights['w_conv3'])
    tf.summary.histogram('biases3',biases['b_conv3'])
    tf.summary.histogram('act3',conv3)

    tf.summary.histogram('weights4',weights['w_conv4'])
    tf.summary.histogram('biases4',biases['b_conv4'])
    tf.summary.histogram('act4',conv4)

    tf.summary.histogram('weights5',weights['w_conv5'])
    tf.summary.histogram('biases5',biases['b_conv5'])
    tf.summary.histogram('act5',conv5)

    tf.summary.histogram('weights6',weights['w_conv6'])
    tf.summary.histogram('biases6',biases['b_conv6'])
    tf.summary.histogram('act6',conv6)

    tf.summary.histogram('weights7',weights['w_conv7'])
    tf.summary.histogram('biases7',biases['b_conv7'])
    tf.summary.histogram('act7',conv7)

    tf.summary.histogram('weights8',weights['w_conv8'])
    tf.summary.histogram('biases8',biases['b_conv8'])
    tf.summary.histogram('act8',conv8)

    tf.summary.histogram('weights9',weights['w_conv9'])
    tf.summary.histogram('biases9',biases['b_conv9'])
    tf.summary.histogram('act9',conv9)

    tf.summary.histogram('weights10',weights['w_conv10'])
    tf.summary.histogram('biases10',biases['b_conv10'])
    tf.summary.histogram('act10',conv10)


print "Training"


def train(x):
    """
    Trains the neural net
    :param x: Features placeholder
    :return: Trained neural net
    """
    prediction = convNeuralNet(x)
    #print prediction
    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
        tf.summary.scalar('cross_entropy',cost)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # learning rate = 0.001

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        tf.summary.scalar('accuracy',accuracy)

    # cycles of feed forward and backprop
    num_epochs = ne

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.getcwd()+tb_path)
        writer.add_graph(sess.graph)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(int(real_X_9.shape[0])/batch_size):#mnist.train.num_examples/batch_size)): # X.shape[0]
                randidx = np.random.choice(real_X_9.shape[0], batch_size, replace=False)
                epoch_x,epoch_y = real_X_9[randidx,:],real_y_9[randidx,:] #mnist.train.next_batch(batch_size) # X,y
                j,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                if i == 0:
                    [ta] = sess.run([accuracy],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                    print 'Train Accuracy', ta
                if epoch % 50 == 0 and i == 0:
                    #saver.save(sess,os.getcwd()+'/models/location/locationCNN18.ckpt')
                    #print 'Checkpoint saved at',os.getcwd()+'/models/location/locationCNN18'
                    pass
                    #ta_list.append(ta)
                if i % 5 == 0:
                    s = sess.run(merged_summary,feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                    writer.add_summary(s,i)

                epoch_loss += c
            print '\n','Epoch', epoch + 1, 'completed out of', num_epochs, '\nLoss:',epoch_loss

        saver.save(sess, os.getcwd() + '/models/location/' + NAME)
        saver.export_meta_graph(os.getcwd() + '/models/location/' + NAME + '.meta')
        print "Model saved"

        print '\n','Train Accuracy', accuracy.eval(feed_dict={x:real_X_9, y:real_y_9, keep_prob:TRAIN_KEEP_PROB})
        print '\n','Test Accuracy', accuracy.eval(feed_dict={x:test_real_X, y:test_real_y, keep_prob:1.0}) #X, y #mnist.test.images, mnist.test.labels

        #saver.save(sess,'baseDNN',global_step=1000)

        '''
        Run this:
        tensorboard --logdir=tensorboard/baseDNN-SPECIFICATIONS --debug
        '''


train(x)
