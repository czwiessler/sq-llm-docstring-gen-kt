'''
Implements the Domain Specific Pipeline (DSP)
Runs an MCTS modified to implement Eterna player strategies
Second process of the SAP
'''

import sys
import numpy as np
from eterna_score import get_pairmap_from_secstruct
import RNA
from subprocess import Popen, PIPE, STDOUT
import re
from difflib import SequenceMatcher
import copy
import math


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


def str_to_num(s):
    if s == 'A':
        return 1
    elif s == 'U':
        return 2
    elif s == 'G':
        return 3
    elif s == 'C':
        return 4


def pairmap_from_sequence(seq, vienna_version, vienna_path='../../../EteRNABot/eternabot/./RNAfold'):
    new_struc = ''
    if vienna_version == 1:
        if sys.version_info[:3] > (3,0):
            p = Popen([vienna_path, '-T','37.0'], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
        else:
            p = Popen([vienna_path, '-T','37.0'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        pair = p.communicate(input=''.join(seq))[0]
        formatted = re.split('\s+| \(?\s?',pair)
        new_struc = formatted[1]
    elif vienna_version == 2:
        new_struc = RNA.fold(''.join(seq))[0]

    return get_pairmap_from_secstruct(new_struc)

dot_bracket = '.....((((..((((....)))).)))).....'
seq_str = 'A'*len(dot_bracket)


def dsp(dot_bracket, seq_str, vienna_version='1', vienna_path='../../../EteRNABot/eternabot/./RNAfold'):  # domain specific pipeline
    '''
    Adds player strategies via a MCTS

    :param dot_bracket: The target structure of the RNA in dot-bracket notation
    :param seq_str: The current RNA sequence
    :param vienna_version: Vienna 1.8.5 or Vienna 2
    :param vienna_path: Path to the Vienna 1.8.5 RNAfold
    :return: The updated RNA sequence after the DSP
    '''

    try:
        vienna_version = int(vienna_version)
    except TypeError:
        raise TypeError('Please pass in a valid Vienna version')
    assert(vienna_version <= 2, "Please pass in a valid Vienna version")
    assert(vienna_version >= 1, "Please pass in a valid Vienna version")

    seq = list(seq_str)
    m = []
    SOLVE = False

    current_struc,_ = RNA.fold(seq_str)
    target_struc = encode_struc(dot_bracket)
    target_pm = get_pairmap_from_secstruct(dot_bracket)
    current_pm = get_pairmap_from_secstruct(current_struc)

    pairs = find_parens(dot_bracket)

    #print target_pm
    #print current_pm


    """
    Correcting incorrect base pairings
    """
    ############ Comment out from here to remove this strategy #############
    for base1, base2 in pairs.items(): # corrects incorrect base pairings
        #print base1,base2
        if (seq[base1] == 'A' and seq[base2] == 'U') or (seq[base1] == 'U' and seq[base2] == 'A'):
            continue
        elif (seq[base1] == 'G' and seq[base2] == 'U') or (seq[base1] == 'U' and seq[base2] == 'G'):
            continue
        elif (seq[base1] == 'G' and seq[base2] == 'C') or (seq[base1] == 'C' and seq[base2] == 'G'):
            continue
        elif (seq[base1] == 'G' and seq[base2] == 'A'):
            seq[base1] = 'U'
            m.append([2,base1+1])
        elif (seq[base1] == 'A' and seq[base2] == 'G'):
            seq[base1] = 'C'
            m.append([4,base1+1])
        elif (seq[base1] == 'C' and seq[base2] == 'U'):
            seq[base1] = 'A'
            m.append([1,base1+1])
        elif (seq[base1] == 'U' and seq[base2] == 'C'):
            seq[base1] = 'G'
            m.append([3,base1+1])
        elif (seq[base1] == 'A' and seq[base2] == 'C'):
            seq[base1] = 'G'
            m.append([3,base1+1])
        elif (seq[base1] == 'C' and seq[base2] == 'A'):
            seq[base1] = 'U'
            m.append([2,base1+1])
        elif (seq[base1] == 'A' and seq[base2] == 'A'):
            seq[base1] = 'U'
            m.append([2,base1+1])
        elif (seq[base1] == 'U' and seq[base2] == 'U'):
            seq[base1] = 'A'
            m.append([1,base1+1])
        elif (seq[base1] == 'G' and seq[base2] == 'G'):
            seq[base1] = 'C'
            m.append([4,base1+1])
        elif (seq[base1] == 'C' and seq[base2] == 'C'):
            seq[base1] = 'G'
            m.append([3,base1+1])

    #print ''.join(seq)

    for i in range(len(target_pm)):
        if target_pm[i] == -1:
            seq[i] = 'A'
            m.append([1,i+1])
        else:
            continue
    ######################################################################

    """
    End pairs to G-C
    """
    ############ Comment out from here to remove this strategy #############
    for i in range(len(dot_bracket)):
        try:
            if dot_bracket[i] == '(':# or dot_bracket[i] == ')':
                #print dot_bracket[i]
                if dot_bracket[i-1] == '.' or dot_bracket[i-1] == ')' or dot_bracket[i+1] == '.' or dot_bracket[i+1] == ')':
                    #print i
                    if (seq[i] == 'G' and seq[target_pm[i]] == 'C') or (seq[i] == 'C' and seq[target_pm[i]] == 'G'):
                        continue
                    else:
                        seq[i] = 'G'
                        seq[target_pm[i]] = 'C'
                        m.append([3,i+1])
                        m.append([4,target_pm[i]+1])

                # elif dot_bracket[i+1] == '.' and dot_bracket[i+2] == '.' and dot_bracket[i+3] == '.' and dot_bracket[i+4] == '.':
                #     seq[i+1] = 'G'

            elif dot_bracket[i] == ')':# or dot_bracket[i] == ')':
                #print dot_bracket[i]
                if dot_bracket[i-1] == '.' or dot_bracket[i-1] == '(' or dot_bracket[i+1] == '.' or dot_bracket[i+1] == '(':
                    #print i
                    if (seq[i] == 'G' and seq[target_pm[i]] == 'C') or (seq[i] == 'C' and seq[target_pm[i]] == 'G'):
                        continue
                    else:
                        seq[i] = 'G'
                        seq[target_pm[i]] = 'C'
                        m.append([3,i+1])
                        m.append([4,target_pm[i]+1])

        except IndexError:
            continue
    ######################################################################

    """
    G External Loop Boost
    """
    ############ Comment out from here to remove this strategy #############
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == '(':
            if dot_bracket[i+1] == '.' and dot_bracket[i+2] == '.' and dot_bracket[i+3] == '.' and dot_bracket[i+4] == '.':
                seq[i+1] = 'G'
                m.append([3,i+2])
            # elif (dot_bracket[i+1] == '.' and dot_bracket[i+2] == '('):
            #     seq[i+1] = 'G'
    ########################################################################

    """
    G-A Internal Loop Boost
    """
    ############ Comment out ######################################################
    for i in range(len(dot_bracket)):
        #pairing = target_pm[i]
        if dot_bracket[i] == '(' and dot_bracket[i+1] == '.':# and dot_bracket[target_pm[i]] == ")" and dot_bracket[target_pm[i-1]] == '.':
            leftdots = []
            starter = 0
            for j in range(i+1, len(dot_bracket)):
                if dot_bracket[j] == '(':# or dot_bracket[j] == ')':
                    starter = j
                    break
                leftdots.append(dot_bracket[j])

            rightdots = []
            idx = target_pm[i]
            ender = 0
            for k in range(idx-1,-1,-1):
                if dot_bracket[k] == ')':# or dot_bracket[k] == '(':
                    ender = k
                    break
                rightdots.append(dot_bracket[k])

            if len(leftdots) > 0 and len(rightdots) > 0:
                if (len(leftdots) != 2 or len(rightdots) != 2) and (math.fabs(len(rightdots) - len(leftdots)) <= 5):
                    if target_pm[starter] == starter or target_pm[starter] == ender:
                        seq[i+1] = 'G'
                        seq[ender+1] = 'G'
    ##############################################################################

        # if dot_bracket[i] == ')' and dot_bracket[i+1] == '.' and dot_bracket[target_pm[i]] == "(" and dot_bracket[target_pm[i+1]] == '.':
        #     dots = []
        #     starter = 0
        #     for j in range(i+1, len(dot_bracket)):
        #         if dot_bracket[j] == ')':
        #             starter = j
        #             break
        #         dots.append(dot_bracket[j])
        #
        #     idx = target_pm[i]
        #     ender = 0
        #     for k in range(idx-1,-1,-1):
        #         if dot_bracket[k] == '(':
        #             ender = k
        #             break
        #         dots.append(dot_bracket[k])
        #
        #     if dots.count(dots[0]) == len(dots) and target_pm[ender] == starter:
        #         seq[i+1] = 'G'
        #         seq[ender+1] = 'G'

        """
        U-G-U-G Superboost
        """
        ############ Comment out from here to remove this strategy #############
        if dot_bracket[i] == '(' and dot_bracket[i+1] == '.' and dot_bracket[i+2] == '.' and dot_bracket[i+3] == '(': # UGUG superboost
            idx = target_pm[i]
            dots = []
            starter = 0
            for j in range(i + 1, len(dot_bracket)):
                if dot_bracket[j] == '(':
                    starter = j
                    break
                dots.append(dot_bracket[j])

            idx = target_pm[i]
            ender = 0
            for k in range(idx - 1, -1, -1):
                if dot_bracket[k] == ')':
                    ender = k
                    break
                dots.append(dot_bracket[k])
            if dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == '.' and dot_bracket[idx-3] == ')' and target_pm[ender] == starter:
                seq[i+1] = 'U'
                seq[i+2] = 'G'
                seq[idx-2] = 'U'
                seq[idx-1] = 'G'
                m.append([2,i+2])
                m.append([3,i+3])
                m.append([2,idx-1])
                m.append([3,idx])

            elif dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == ')':
                seq[i+1] = 'G'
                seq[idx-1] = 'G'
                m.append([3,i+2])
                m.append([3,idx])
        ######################################################################
        # if dot_bracket[i] == '(' and dot_bracket[i+1] == '.' and dot_bracket[i+2] == '(': # G-G in 2 pair internal loop
        #     idx = target_pm[i]
        #     if dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == ')':
        #         seq[i+1] = 'G'
        #         seq[idx-1] = 'G'
        #         m.append([3,i+2])
        #         m.append([3,idx])
        #     elif dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == '.' and dot_bracket[idx-3] == ')':
        #         seq[i+1] = 'G'
        #         seq[idx-1] = 'G'
        #         m.append([3,i+2])
        #         m.append([3,idx])
        #
        # if dot_bracket[i] == '(' and dot_bracket[i+1] == '.' and dot_bracket[i+2] == '.' and dot_bracket[i+3] == '.' and dot_bracket[i+4] == '(': # G-G in 2 pair internal loop
        #     idx = target_pm[i]
        #     if dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == '.' and dot_bracket[idx-3] == '.' and dot_bracket[idx-4] == ')':
        #         seq[i+1] = 'G'
        #         seq[idx-3] = 'G'
        #         m.append([3,i+2])
        #         m.append([3,idx])
            # elif dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == '.' and dot_bracket[idx-3] == ')':
            #     seq[i+1] = 'G'
            #     seq[idx-1] = 'G'
            #     m.append([3,i+2])
            #     m.append([3,idx])


    '''
    Flips base pairs
    '''
    ############ Comment out from here to remove this strategy #############
    new_pm = pairmap_from_sequence(seq, vienna_version)
    match = SequenceMatcher(None,new_pm,target_pm).ratio()
    for j in range(5):
        for i in range(len(dot_bracket)):
            if new_pm == target_pm:
                print('puzzle solved')
                SOLVE = True
                break
            else:
                if new_pm[i] == target_pm[i]:
                    continue
                else:
                    paired = target_pm[i]
                    base1 = seq[i]
                    base2 = seq[paired]

                    if paired == -1: continue

                    seq[i] = base2
                    seq[paired] = base1
                    new_pm = pairmap_from_sequence(seq, vienna_version)

                    new_match = SequenceMatcher(None,new_pm,target_pm).ratio()

                    if new_match > match:
                        match = copy.deepcopy(new_match)
                        m.append([str_to_num(seq[i]),i+1])
                        m.append([str_to_num(seq[paired]),paired+1])
                    else:
                        seq[i] = base1
                        seq[paired] = base2
    ######################################################################

    for i in range(len(dot_bracket)):
        if new_pm[i] == target_pm[i]:
            continue
        else:
            paired = target_pm[i]
            seq[i] = 'G'
            seq[paired] = 'C'
            m.append([3,i+1])
            m.append([4,paired+1])

    for j in range(3):
        for i in range(len(dot_bracket)):
            if new_pm == target_pm:
                print('puzzle solved')
                SOLVE = True
                break
            else:
                if new_pm[i] == target_pm[i]:
                    continue
                else:
                    paired = target_pm[i]
                    base1 = seq[i]
                    base2 = seq[paired]

                    if paired == -1: continue

                    seq[i] = base2
                    seq[paired] = base1
                    new_pm = pairmap_from_sequence(seq, vienna_version)

                    new_match = SequenceMatcher(None,new_pm,target_pm).ratio()

                    if new_match > match:
                        match = copy.deepcopy(new_match)
                        m.append([str_to_num(seq[i]),i+1])
                        m.append([str_to_num(seq[paired]),paired+1])
                    else:
                        seq[i] = base1
                        seq[paired] = base2

    return ''.join(seq),m,SOLVE

    cs,_ = RNA.fold(''.join(seq))
    current_pm = get_pairmap_from_secstruct(cs)

# print current_pm
# print target_pm
#print dsp(dot_bracket,seq_str)
