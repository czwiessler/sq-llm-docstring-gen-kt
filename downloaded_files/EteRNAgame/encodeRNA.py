# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:51:30 2017

@author: Rohan
"""

'''
encode RNA strucutre and encode movesets
'''

import copy
from getData import getStructure, getTargetEnergy
from readData import read_structure, read_locks, read_structure_raw, format_pairmap

# def longest(a):
#     return max(len(a), * map(longest, a)) if isinstance(a, list) and a else 0

def base_sequence_at_current_time_pr(ms,struc):
    '''
    For every move change, updates the complete sequence of the RNA, updated
    to support pasting and resetting

    :param ms: The encoded move sets
    :param struc: Encoded structure

    :return: List of updated base sequences for every base change
    '''
    #Z = []
    Z = []
    # for i in ms:
    #     Z = []
    #     for j in range(0,len(i)):
    #         temp = copy.deepcopy(struc[ms.index(i)]) # copy.deepcopy()
    #         print 'deep copy',temp
    #         Z.append(temp)
    #     print 'copied thing',Z
    #     Z1.append(Z)
    # print "Z1",Z1

    for ctr in range(len(struc)):
        Z1 = []
        temp = copy.deepcopy(struc[ctr])
        Z1.append(temp)
        #print 'Initial result_list', Z1

        j = 0
        for i in range(len(ms[ctr])-1): # 0,len(ms[ctr])-1
        #Z = []
        #for j in range(len(i)):
            try:
                if ms[ctr][i][0] == 'paste' or ms[ctr][i][0] == 'reset':
                    temp = copy.deepcopy(Z1[j])
                    pasted = copy.deepcopy(ms[ctr][i][1])
                    temp[:len(pasted)] = pasted
                    Z1.append(temp)
                else:
                    #print 'Z1[j]=',Z1[j]
                    temp = copy.deepcopy(Z1[j]) # copy.deepcopy()VR
                    #print 'temp=',temp
                    base = ms[ctr][i][0]
                    loc = ms[ctr][i][1]-1
                    #loc = ms[Z1.index(i)][j][1] - 1
                    #print 'location=',loc
                    #print 'base=',base
                    temp[loc] = base
                    #print 'temp after changing=',temp
                    Z1.append(temp)
                    #print 'end',Z1
            except IndexError:
                continue
            j = j + 1
        Z.append(Z1)

    #print 'Leaving function're
    return Z

'''
DEPRECATED version of base_sequence_at_current_time_pr
'''
def base_sequence_at_current_time(ms,struc):
    #Z = []
    Z = []
    # for i in ms:
    #     Z = []
    #     for j in range(0,len(i)):
    #         temp = copy.deepcopy(struc[ms.index(i)]) # copy.deepcopy()
    #         print 'deep copy',temp
    #         Z.append(temp)
    #     print 'copied thing',Z
    #     Z1.append(Z)
    # print "Z1",Z1

    for ctr in range(len(struc)):
        Z1 = []
        temp = copy.deepcopy(struc[ctr])
        Z1.append(temp)
        #print 'Initial result_list', Z1

        j = 0
        for i in range(0,len(ms[ctr])-1):
        #Z = []
        #for j in range(len(i)):
            try:
                #print 'Z1[j]=',Z1[j]
                temp = copy.deepcopy(Z1[j]) # copy.deepcopy()VR
                #print 'temp=',temp
                base = ms[ctr][i][0]
                loc = ms[ctr][i][1]-1
                #loc = ms[Z1.index(i)][j][1] - 1
                #print 'location=',loc
                #print 'base=',base
                temp[loc] = base
                #print 'temp after changing=',temp
                Z1.append(temp)
                #print 'end',Z1
            except IndexError:
                continue
            j = j + 1
        Z.append(Z1)

    #print 'Leaving function'
    return Z

'''
Another DEPRECATED version of base_sequence_at_current_time_pr
'''
def base_sequence_at_current_time_deprecated(ms,struc):
    Z = []
    Z1 = []
    for i in ms:
        Z = []
        for j in range(0,len(i)):
            temp = copy.deepcopy(struc[ms.index(i)])
            Z.append(temp)
        #print Z
        Z1.append(Z)

    y0 = []
    for i in ms:
        for j in i:
            y0.append(j)

    for i in (Z1):
        for j in range(len(i)):
            try:
                #print j
                loc = ms[Z1.index(i)][j][1] - 1
                #print loc
                #print 'index',i[j+1][loc]
                #print i[j+1]
                i[j+1][loc] = ms[Z1.index(i)][j][0]
                #print Z1
            except IndexError:
                continue

    return Z1


def structure_and_energy_at_current_time(base_seq,pid):
    '''
    Calculates the natural structure and energy of RNA sequences and generates
    complete featurespace for the CNNs

    :param base_seq: The current RNA sequence
    :param pid: The puzzle ID
    :return: The full feature set for the CNNs
                [base sequence, current structure, target structure, current energy
                    target energy, current pairmap, target pairmap, locked bases]
    '''
    Z2 = []
    for i in base_seq:
        for j in i:
            struc,energy = (getStructure(j))
            current_pairmap = format_pairmap(struc)
            raw_struc = read_structure_raw(pid)
            target_pairmap = format_pairmap(raw_struc)
            enc_struc = []
            for k in struc:
                if k == '.':
                    enc_struc.append(1)
                elif k == '(':
                    enc_struc.append(2)
                elif k == ')':
                    enc_struc.append(3)
            target = read_structure(pid)
            len_puzzle = len(target)
            target_energy = getTargetEnergy(j,target)
            locks = read_locks(pid)
            if locks == "None":
                locks = [1]*len_puzzle
            attrs = [j,enc_struc,target,energy,target_energy,current_pairmap,target_pairmap,locks]
            #attrs = [j,current_pairmap,target_pairmap,energy,target_energy,locks]
            Z2.append(attrs)

    return Z2


def structure_and_energy_at_current_time_with_location(base_seq,pid,moveset,longest):
    '''
    The same program as above with location added as a feature to pass to baseCNN

    :param base_seq: The current RNA sequence
    :param pid: The puzzle ID
    :param moveset: The move sets
    :param longest: The length of the longest puzzle in the featurespace
    :return: The full feature set for the CNNs
                [base sequence, current structure, target structure, current energy
                    target energy, current pairmap, target pairmap, locked bases]
    '''
    Z2 = []
    for i in base_seq:
        for j in i:
            struc,energy = (getStructure(j))
            enc_struc = []
            for k in struc:
                if k == '.':
                    enc_struc.append(1)
                elif k == '(':
                    enc_struc.append(2)
                elif k == ')':
                    enc_struc.append(3)
            target = read_structure(pid)
            len_puzzle = len(target)
            target_energy = getTargetEnergy(j,target)
            locks = read_locks(pid)
            if locks == "None":
                locks = [1]*len_puzzle
            location = encode_location(moveset,longest)
            attrs = [j,enc_struc,target,energy,target_energy,locks,location]
            Z2.append(attrs)

    return Z2

'''
DEPRECATED
'''
def encode_movesets(moveset):
    ms = []
    #lens = [len(x) for j in x for x in moveset]
    #max_lens = max(lens)
    for k in moveset:
        player = []
        for i in k:
            for j in i:
                if 'type' in j:
                    continue
                    # player.append(1)
                    # player.append(12345)
                elif j['base'] == 'A':
                    player.append(1)
                    player.append(j['pos'])
                elif j['base'] == 'U':
                    player.append(2)
                    player.append(j['pos'])
                elif j['base'] == 'G':
                    player.append(3)
                    player.append(j['pos'])
                elif j['base'] == 'C':
                    player.append(4)
                    player.append(j['pos'])
                elif j['type'] == 'paste' or j['type'] == 'reset':
                    continue
        ms.append(player)
    '''
    lens = [len(j) for j in ms]
    max_lens = max(lens)
    #ms2 = []

    for l in ms:
        l.extend([None]*(max_lens-len(l)))
    '''

    return ms

'''
DEPRECATED
'''
def encode_movesets_style(moveset):
    ms = []
    #lens = [len(x) for j in x for x in moveset]
    #max_lens = max(lens)
    for k in moveset:
        player = []
        for i in k:
            for j in i:
                if 'type' in j:
                    player.append([1,1])
                elif j['base'] == 'A':
                    player.append([1,j['pos']])
                elif j['base'] == 'U':
                    player.append([2,j['pos']])
                elif j['base'] == 'G':
                    player.append([3,j['pos']])
                elif j['base'] == 'C':
                    player.append([4,j['pos']])
                elif j['type'] == 'paste' or j['type' == 'reset']:
                    player.append([1,1])

        ms.append(player)
    lens = [len(j) for j in ms]
    max_lens = max(lens)
    #ms2 = []
    '''
    for l in ms:
        l.extend([None]*(max_lens-len(l)))
    '''

    return ms


def encode_movesets_style_pr(moveset):
    '''
    Encodes the raw move set data into [base,location]
    Updated to support pasting and resetting

    :param moveset: The move set list
    :return: List of lists containing [base,location]
    '''
    ms = []
    #lens = [len(x) for j in x for x in moveset]
    #max_lens = max(lens)
    for k in moveset:
        player = []
        for i in k:
            for j in i:
                if 'type' in j:
                    if j['type'] == 'paste': #player.append([1,1]) # FIX THIS URGENT
                        seqlist = []
                        for a in j['sequence']: #player.append([1,1]) #continue #FIX
                            if a == 'A':
                                seqlist.append(1)
                            if a == 'U':
                                seqlist.append(2)
                            if a == 'G':
                                seqlist.append(3)
                            if a == 'C':
                                seqlist.append(4)
                        player.append(['paste',seqlist])
                    elif j['type'] == 'reset':
                        seqlist = []
                        for a in j['sequence']: #player.append([1,1]) #continue #FIX
                            if a == 'A':
                                seqlist.append(1)
                            if a == 'U':
                                seqlist.append(2)
                            if a == 'G':
                                seqlist.append(3)
                            if a == 'C':
                                seqlist.append(4)
                        player.append(['reset',seqlist])
                elif j['base'] == 'A':
                    player.append([1,j['pos']])
                elif j['base'] == 'U':
                    player.append([2,j['pos']])
                elif j['base'] == 'G':
                    player.append([3,j['pos']])
                elif j['base'] == 'C':
                    player.append([4,j['pos']])
                elif j['type'] == 'paste' or j['type'] == 'reset':
                    seqlist = []
                    for a in j['sequence']: #player.append([1,1]) #continue #FIX
                        if a == 'A':
                            seqlist.append(1)
                        if a == 'U':
                            seqlist.append(2)
                        if a == 'G':
                            seqlist.append(3)
                        if a == 'C':
                            seqlist.append(4)
                    player.append(['paste',seqlist])
        ms.append(player)
    #lens = [len(j) for j in ms]
    #max_lens = max(lens)
    #ms2 = []
    '''
    for l in ms:
        l.extend([None]*(max_lens-len(l)))
    '''

    return ms

'''
DEPRECATED
'''
def encode_movesets_style_dev(moveset):
    ms = []
    #lens = [len(x) for j in x for x in moveset]
    #max_lens = max(lens)
    for k in moveset:
        #player = []
        for i in k:
            for j in i:
                if 'type' in j:
                    ms.append([0,0]) # FIX THIS URGENT
                elif j['base'] == 'A':
                    ms.append([1,j['pos']])
                elif j['base'] == 'U':
                    ms.append([2,j['pos']])
                elif j['base'] == 'G':
                    ms.append([3,j['pos']])
                elif j['base'] == 'C':
                    ms.append([4,j['pos']])
                elif j['type'] == 'paste' or j['type'] == 'reset':
                    continue
        #ms.append(player)
    lens = [len(j) for j in ms]
    max_lens = max(lens)
    #ms2 = []
    '''
    for l in ms:
        l.extend([None]*(max_lens-len(l)))
    '''

    return ms

def encode_bases(moveset):
    '''
    Encodes the labels for baseCNN

    :param moveset: The move set list
    :return: One-hot encoded nucleotides
                [A, U, G, C]
    '''
    ms = []
    #lens = [len(x) for j in x for x in moveset]
    #max_lens = max(lens)
    for k in moveset:
        player = []
        if not k:
            ms.append([1,1,1,1])
        else:
            for i in k:
                for j in i:
                    if 'type' in j:
                        ms.append([1,1,1,1])#continue #player.append([0,0])
                    elif j['base'] == 'A':
                        ms.append([1,0,0,0])
                    elif j['base'] == 'U':
                        ms.append([0,1,0,0])
                    elif j['base'] == 'G':
                        ms.append([0,0,1,0])
                    elif j['base'] == 'C':
                        ms.append([0,0,0,1])
                    elif j['type'] == 'paste' or j['type'] == 'reset':
                        ms.append([1,1,1,1])
        #ms.append(player)
    #lens = [len(j) for j in ms]
    #max_lens = max(lens)
    #ms2 = []
    '''
    for l in ms:
        l.extend([None]*(max_lens-len(l)))
    '''

    return ms


def encode_location(moveset,puzzle_length):
    '''
    Encodes the labels for locationCNN

    :param moveset: The move set list
    :return: One-hot encoded location vectors
    '''
    ms = []
    #lens = [len(x) for j in x for x in moveset]
    #max_lens = max(lens)
    for k in moveset:
        player = []
        if not k:
            num_bases = puzzle_length
            loc_list = [1] * num_bases
            ms.append(loc_list)
        else:
            for i in k:
                for j in i:
                    if 'type' in j:
                        num_bases = puzzle_length
                        loc_list = [1] * num_bases
                        ms.append(loc_list)
                    # elif j['type'] == 'paste' or j['type'] == 'reset':
                    #     num_bases = puzzle_length
                    #     loc_list = [1] * num_bases
                    #     ms.append(loc_list)
                    else:
                        loc = int(j['pos']) - 1
                        num_bases = puzzle_length # this is the number of classes for tf DNN/RNN
                        loc_list = [0] * (num_bases - 1)
                        #loc_list[loc] = 1
                        loc_list.insert(loc,1)
                        ms.append(loc_list)
        #ms.append(player)
    # lens = [len(j) for j in ms]
    # max_lens = max(lens)
    #ms2 = []
    '''
    for l in ms:
        l.extend([None]*(max_lens-len(l)))
    '''

    return ms

'''
DEPRECATED
'''
def encode_structure(structure):
  encoded_structure = []
  for i in structure:
    if i == ".":
      encoded_structure.append(0)
    elif i == "(" or i == ")":
      encoded_structure.append(1)

  return encoded_structure

#########################################################################

# def encode_movesets(moveset):
#   encoded_ms = []
#   for move in moveset:
#     if move[0]['base'] == 'A':
#       encoded_ms.append([1,move[0]['pos']])
#     elif move[0]['base'] == 'U':
#       encoded_ms.append([2,move[0]['pos']])
#     elif move[0]['base'] == 'G':
#       encoded_ms.append([3,move[0]['pos']])
#     elif move[0]['base'] == 'C':
#       encoded_ms.append([4,move[0]['pos']])
#
#   return encoded_ms
# '''
# '''
# l = []
# movesets = [[{'base':'G','pos':3}],[{'base':'A','pos':8},{'base':'U','pos':12}]]
# for move in movesets:
#   for i in move:
#     pass
#
# def encode_movesets_v0(moveset):
#     ms = []
#     max_moves = longest(moveset)
#     for k in moveset:
#         #max_moves = len(max(k,key=len))
#         #max_moves = longest(k)
#         soln = []
#         #if k[0][0]['type'] == 'reset':
#          #   data_6892344.pop(data_6892344.index(k))
#         for i in k:
#             n_moves = len(i)
#             i_moves = []
#             for j in i:
#                 if 'type' in j:
#                     i_moves.append([1,12345])
#                 elif j['base'] == 'A':
#                     i_moves.append([1,j['pos']])
#                 elif j['base'] == 'U':
#                     i_moves.append([2,j['pos']])
#                 elif j['base'] == 'G':
#                     i_moves.append([3,j['pos']])
#                 elif j['base'] == 'C':
#                     i_moves.append([4,j['pos']])
#                 elif j['type'] == 'paste' or j['type'] == 'reset':
#                     continue
#             #i_moves = [i_moves]
#             i_moves = i_moves + (max_moves - n_moves)*[[0,0]]
#             soln.append(i_moves)
#         ms.append(soln)
#
#     return ms
# '''
# '''
# def encode_movesets(moveset):
#     ms = []
#     #lens = [len(x) for j in x for x in moveset]
#     #max_lens = max(lens)
#     for k in moveset:
#         player = []
#         for i in k:
#             for j in i:
#                 if 'type' in j:
#                     player.append(1)
#                     player.append(12345)
#                 elif j['base'] == 'A':
#                     player.append(1)
#                     player.append(j['pos'])
#                 elif j['base'] == 'U':
#                     player.append(2)
#                     player.append(j['pos'])
#                 elif j['base'] == 'G':
#                     player.append(3)
#                     player.append(j['pos'])
#                 elif j['base'] == 'C':
#                     player.append(4)
#                     player.append(j['pos'])
#                 elif j['type'] == 'paste' or j['type'] == 'reset':
#                     continue
#         ms.append(player)
#     lens = [len(j) for j in ms]
#     max_lens = max(lens)
#     ms2 = []
#     for l in ms:
#         l.extend([0]*(max_lens-len(l)))
# '''
#
# #    return ms
#
# '''
# data_6892344 = read_movesets(os.getcwd() + '/movesets/move-set-11-14-2016.txt',6892344)
# print len(data_6892344)
# lens = [len(x) for x in j for j in data_6892344]
# print lens
# print sum(lens)
#
# def encode_movesets_dataframe(moveset):
#
#     pass
#
# columns = ['pid','time','base','loc']
# edf = pd.DataFrame(index=range(sum(lens)),columns=columns,dtype='float')
# print edf
# '''
# '''
# encoded_ms = []
# for i in ms_6503049:
#   print (i[0]['pos'])
#   if i[0]['base'] == 'A':
#     encoded_ms.append([1,i[0]['pos']])
#   elif i[0]['base'] == 'U':
#     encoded_ms.append([2,i[0]['pos']])
#   elif i[0]['base'] == 'G':
#     encoded_ms.append([3,i[0]['pos']])
#   elif i[0]['base'] == 'C':
#     encoded_ms.append([4,i[0]['pos']])
#
# print encoded_ms
#
