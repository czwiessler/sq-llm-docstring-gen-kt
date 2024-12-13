import tensorflow as tf
import os
import numpy as np
import RNA
import copy
from numpy.random import choice
from difflib import SequenceMatcher
from readData import format_pairmap

dot_bracket = '.....((((((((((((....))))((((....))))))))((((((((....))))((((....))))))))))))....................'
TF_SHAPE = 6 * 350
BASE_SHAPE = 7 * 350
len_puzzle = len(dot_bracket)
len_longest = 350

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def encode_struc(dots):
    s = []
    for i in dots:
        if i == '.':
            s.append(1)
        elif i == '(':
            s.append(2)
        elif i == ')':
            s.append(3)
    return s

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None

def convert_to_list(base_seq):
    str_struc = []
    for i in base_seq:
        if i == 'A':
            str_struc.append(1)
        elif i == 'U':
            str_struc.append(2)
        elif i == 'G':
            str_struc.append(3)
        elif i == 'C':
            str_struc.append(4)
    #struc = ''.join(str_struc)
    return str_struc

bs = 'A'*len_puzzle
base_seq = (convert_to_list(bs)) + ([0]*(len_longest - len_puzzle))
# cdb = '.'*len_puzzle
current_struc = (encode_struc(RNA.fold(bs)[0])) + ([0]*(len_longest - len_puzzle))
target_struc = encode_struc(dot_bracket) + ([0]*(len_longest - len_puzzle))
current_energy = [0.0] + ([0]*(len_longest - 1))
target_energy = [0.0] + ([0]*(len_longest - 1))
locks = ([1]*len_puzzle) + ([0]*(len_longest - len_puzzle))

print len(base_seq),len(current_struc),len(dot_bracket),len(target_struc),len(current_energy),len(target_energy),len(locks)

inputs2 = np.array([base_seq,current_struc,target_struc,current_energy,target_energy,locks])
inputs = inputs2.reshape([-1,TF_SHAPE])

with tf.Graph().as_default() as base_graph:
    saver1 = tf.train.import_meta_graph(os.getcwd()+'/models/base/baseCNN7.meta')
sess1 = tf.Session(graph=base_graph) # config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
saver1.restore(sess1,os.getcwd()+'/models/base/baseCNN7')

x = base_graph.get_tensor_by_name('x_placeholder:0')
y = base_graph.get_tensor_by_name('y_placeholder:0')
keep_prob = base_graph.get_tensor_by_name('keep_prob_placeholder:0')

base_weights = base_graph.get_tensor_by_name('op7:0')

base_feed_dict={x:inputs,keep_prob:1.0}

with tf.Graph().as_default() as location_graph:
    saver2 = tf.train.import_meta_graph(os.getcwd()+'/models/location/locationCNN7.meta')
sess2 = tf.Session(graph=location_graph) # config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
saver2.restore(sess2,os.getcwd()+'/models/location/locationCNN7')

x2 = location_graph.get_tensor_by_name('x_placeholder:0')
y2 = location_graph.get_tensor_by_name('y_placeholder:0')
keep_prob2 = location_graph.get_tensor_by_name('keep_prob_placeholder:0')

location_weights = location_graph.get_tensor_by_name('op7:0')

print 'models loaded'

location_feed_dict = {x2:inputs,keep_prob2:1.0}
movesets = []
iteration = 0
for i in range(1000):
    if np.all(inputs2[1] == inputs2[2]):
        print("Puzzle Solved")
        break
    else:
        location_array = ((sess2.run(location_weights,location_feed_dict))[0])

        inputs2 = inputs.reshape([6,TF_SHAPE/6])
        location_array = location_array[:len_puzzle] - min(location_array[:len_puzzle])
        total_l = sum(location_array)
        location_array = location_array/total_l
        location_change = (choice(list(range(0,len(location_array))),1,p=location_array,replace=False))[0]
        #location_change = np.argmax(location_array)
        la = [0.0] * len_longest
        la[location_change] = 1.0
        inputs2 = np.append(inputs2, la)
        inputs = inputs2.reshape([-1,BASE_SHAPE])
        base_feed_dict = {x:inputs,keep_prob:1.0}

        base_array = ((sess1.run(base_weights,base_feed_dict))[0])
        base_array = base_array - min(base_array)

        total = sum(base_array)
        base_array = base_array/total

        #if np.random.rand() > 0.0:
        # FOR CHOOSING STOCHASTICALLY
        base_change = (choice([1,2,3,4],1,p=base_array,replace=False))[0]
        #else:
        # NOT STOCHASTICALLY
        #base_change = np.argmax(base_array) + 1

        inputs2 = inputs.reshape([7,BASE_SHAPE/7])

        # if inputs2[0][location_change] == base_change:
        #     second = second_largest(base_array)
        #     base_change = np.where(base_array==second)[0][0] + 1

        temp = copy.deepcopy(inputs2[0])
        temp[location_change] = base_change
        move = [base_change,location_change]
        movesets.append(move)
        #print move
        str_seq = []
        for i in temp:
            if i == 1:
                str_seq.append('A')
            elif i == 2:
                str_seq.append('U')
            elif i == 3:
                str_seq.append('G')
            elif i == 4:
                str_seq.append('C')
            else:
                continue
        str_seq = ''.join(str_seq)
        str_struc,current_e = RNA.fold(str_seq)
        print str_struc
        #print len(str_struc)
        print similar(str_struc,dot_bracket)
        rna_struc = []
        for i in inputs2[2]:
            if i == 1:
                rna_struc.append('.')
            elif i == 2:
                rna_struc.append('(')
            elif i == 3:
                rna_struc.append(')')
            else:
                continue
        rna_struc = ''.join(rna_struc)
        target_e = RNA.energy_of_structure(str_seq,rna_struc,0)
        enc_struc = []
        for i in str_struc:
            if i == '.':
                enc_struc.append(1)
            elif i == '(':
                enc_struc.append(2)
            elif i == ')':
                enc_struc.append(3)
            else:
                continue
        inputs2[0] = temp
        inputs2[1][:len(enc_struc)] = (enc_struc)
        inputs2[3][0] = current_e
        inputs2[4][0] = target_e
        inputs_loc = inputs2[0:6]
        inputs = inputs_loc.reshape([-1,TF_SHAPE])
        base_feed_dict={x:inputs,keep_prob:1.0}
        location_feed_dict = {x2:inputs,keep_prob2:1.0}
        iteration += 1
        reg = []
        for i in inputs2[0]:
            if i == 1:
                reg.append('A')
            elif i == 2:
                reg.append('U')
            elif i == 3:
                reg.append('G')
            elif i == 4:
                reg.append('C')
            else:
                continue
        reg = ''.join(reg)
        #print inputs2[0][:len_puzzle]
        print reg
        print iteration
        #print current_struc[:len(enc_struc)]
        #print target_struc[:len(enc_struc)]
        #print inputs2[1][:len(enc_struc)]
        #print format_pairmap(str_struc)
        if similar(str_struc,dot_bracket) >= 0.7:
            print 'max'
            print str_struc
            print dot_bracket
            print reg
            break
