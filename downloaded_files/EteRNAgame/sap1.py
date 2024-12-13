'''
Implements the Single Base Mutator (SBC)
Runs a traditional MCTS which mutates single bases and calculates immediate reward
First process of the SAP
'''

import copy
import sys
import re
import RNA
#from eterna_score import eternabot_score
from difflib import SequenceMatcher
#from eterna_score import eternabot_score
from random import choice
from eterna_score import get_pairmap_from_secstruct
from subprocess import Popen, PIPE, STDOUT

vienna_path='../../../EteRNABot/eternabot/./RNAfold'

def fold(seq, vienna_version=1, vienna_path='../../../EteRNABot/eternabot/./RNAfold'):
    if vienna_version == 1:
        if sys.version_info[:3] > (3,0):
            p = Popen([vienna_path, '-T','37.0'], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
        else:
            p = Popen([vienna_path, '-T','37.0'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        pair = p.communicate(input=''.join(seq))[0]
        formatted = re.split('\s+| \(?\s?',pair)
        s = formatted[1]
    else:
        s, _ = RNA.fold(seq)
    return s
    

def hot_one_state(seq,index,base):
    #array = np.zeros(NUM_STATES)
    copied_seq = copy.deepcopy(seq)
    copied_seq[index] = base
    return copied_seq


def convert_to_struc(base_seq, vienna_version=1):
    str_struc = []
    for i in base_seq:
        if i == 1:
            str_struc.append('A')
        elif i == 2:
            str_struc.append('U')
        elif i == 3:
            str_struc.append('G')
        elif i == 4:
            str_struc.append('C')
    struc = ''.join(str_struc)
    s = fold(struc, vienna_version, vienna_path)
    return_struc = []
    for i in s:
        if i == '.':
            return_struc.append(1)
        elif i == '(':
            return_struc.append(2)
        elif i == ')':
            return_struc.append(3)

    return s


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


def convert_to_str(base_str):
    str_struc = []
    for i in base_str:
        if i == 1:
            str_struc.append('A')
        if i == 2:
            str_struc.append('U')
        if i == 3:
            str_struc.append('G')
        if i == 4:
            str_struc.append('C')

    return ''.join(str_struc)


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


def one_hot_seq(seq):
    onehot = []
    for base in seq:
        if base == 1:
            onehot.append([1.,0.,0.,0.])
        elif base == 2:
            onehot.append([0.,1.,0.,0.])
        elif base == 3:
            onehot.append([0.,0.,1.,0.])
        elif base == 4:
            onehot.append([0.,0.,0.,1.])

    return onehot


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_parens(s):
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == '(':
            pstack.append(i)
        elif c == ')':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret


def all_same(items):
    return all(x == items[0] for x in items)


len_longest = 108

# current = convert(base_seq)
dot_bracket = '.....((((((((...((((((((((........))))))))))...((((((((((........))))))))))...((((((((((........))))))))))...)))))))).....'
seq = 'AAAAAGUUUUGAGAAAGAAGUCUGGGGAAAAAAACUUGGGUUUCAAAGGGUGAAAUGGAAAAAAACAUUUCACCCAAAGUUCCUAUCCGAAAAAAAGGAUAGGAGCAAACUUAAAACAAAAA'


def sbc(dot_bracket,seq, vienna_version=1, vienna_path='../../../EteRNABot/eternabot/./RNAfold'): # Monte Carlo Tree Search with Depth 1
    '''
    Mutates individual bases with MCTS

    :param dot_bracket: The target structure for the RNA in dot-bracket notation
    :param seq: The current RNA sequence
    :return: An updated RNA sequence after the SBC
    '''

    movesets = []
    target_struc = encode_struc(dot_bracket)
    pm = get_pairmap_from_secstruct(dot_bracket)
    SOLVE = False
    #cdb = '.((((....))))'
    #current_struc = encode_struc(cdb)
    #percent_match = similar(dot_bracket,cdb)
    len_puzzle = len(target_struc)
    len_puzzle_float = len(target_struc) * 1.0
    # GAACGCACCUGCCUGUUUGGGGAGUAUGAA   GAACGCACCUGCCUGUUUGGGUAGCAUGAA
    # GAACGCACCUGCCUGUCUGGGUAGCAUGAA  GAACUCACCUGCCUGUCUUGGUAGCAUCAA
    # global current_seq
    current_seq = convert_to_list(seq)
    cdb = fold(seq, vienna_version, vienna_path)
    current_pm = get_pairmap_from_secstruct(cdb)
    for i in range(4):
        for location in range(len_puzzle):
            if dot_bracket == cdb:
                # print current_seq
                # print convert_to_str(current_seq)
                # print 'Puzzle Solved'
                SOLVE = True
                break
            else:
                #percent_match = similar(dot_bracket,cdb)

                A = 1
                U = 2
                G = 3
                C = 4

                a_change = hot_one_state(current_seq,location,A)
                u_change = hot_one_state(current_seq,location,U)
                g_change = hot_one_state(current_seq,location,G)
                c_change = hot_one_state(current_seq,location,C)

                a_struc = get_pairmap_from_secstruct(convert_to_struc(a_change))
                u_struc = get_pairmap_from_secstruct(convert_to_struc(u_change))
                g_struc = get_pairmap_from_secstruct(convert_to_struc(g_change))
                c_struc = get_pairmap_from_secstruct(convert_to_struc(c_change))
                # print '\n'
                # # print a_struc
                # # print u_struc
                # # print g_struc
                # # print c_struc
                # print dot_bracket

                # a_reward = SequenceMatcher(None,a_struc,pm)
                # u_reward = SequenceMatcher(None,u_struc,pm)
                # g_reward = SequenceMatcher(None,g_struc,pm)
                # c_reward = SequenceMatcher(None,c_struc,pm)

                a_reward = similar(a_struc,pm)
                u_reward = similar(u_struc,pm)
                g_reward = similar(g_struc,pm)
                c_reward = similar(c_struc,pm)

                # a_reward = eternabot_score(a_struc)
                # u_reward = eternabot_score(u_struc)
                # g_reward = eternabot_score(g_struc)
                # c_reward = eternabot_score(c_struc)
                # print a_reward,u_reward,g_reward,c_reward

                changes = [a_reward,u_reward,g_reward,c_reward]
                # total = sum(changes)
                # base_array = [x / total for x in changes]
                # best_move = (choice([1,2,3,4],1,p=base_array,replace=False))[0]
                if all_same(changes) != True:
                    indices = [i for i, x in enumerate(changes) if x == max(changes)]
                    best_move = choice(indices) + 1
                    new_seq = hot_one_state(current_seq,location,best_move)

                    current_seq = new_seq
                    cdb = convert_to_struc(current_seq)
                    m = [best_move,location]
                    movesets.append(m)
                else:
                    pass
                # print current_seq
                reg = []
                # print convert_to_str(current_seq)
                # print cdb
                current_pm = get_pairmap_from_secstruct(cdb)
                # print len(list(set(current_pm) & set(pm)))/float((len(pm)))

    return convert_to_str(current_seq),movesets,SOLVE

#print sbc(dot_bracket,seq)
