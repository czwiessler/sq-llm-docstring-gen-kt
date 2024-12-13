# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:18:56 2017

@author: rohankoodli
"""

import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import pickle
from readData import read_movesets_uid_pid, experience
import seaborn; seaborn.set()

def plot_moves():
    # filepath = os.getcwd() + '/movesets/move-set-11-14-2016.txt'
    new_ms = os.getcwd() + '/movesets/moveset6-22a.txt'
    # ms_df = pd.read_csv(filepath, sep=" ", header="infer", delimiter='\t')

    #print ms_df

    moveset_dataFrame = pd.read_csv(new_ms, sep=' ', header='infer', delimiter='\t')
    pidList = [6502997,6502995,6502990,6502996,6502963,6502964,6502966,6502967,6502968,6502969,6502970,6502976]
    with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    content = [int(x) for x in content]
    progression = [6502963,6502966,6502968,6502969,6502970,6502973,6502976,6502984,6502985, \
                    6502994,6502995,6502996,6502997,6502998,6502999,6503000] # 6502957
    content.extend(progression)
    l = range(501)

    total = {}
    for pid in content:
        puzzles_pid = moveset_dataFrame.loc[moveset_dataFrame['pid'] == pid]
        plist = list(puzzles_pid['move_set'])
        plist_dict = []
        for i in plist:
            s1 = (ast.literal_eval(i))
            s2 = int(s1['num_moves'])
            plist_dict.append(s2)
        total[str(pid)] = plist_dict
        print('done with pid %i' % pid)
        plt.hist(total[str(pid)],bins=l[::10])

    #plt.savefig('/Users/rohankoodli/Desktop/allsinglestate.png')
    #pickle.dump(total, open(os.getcwd()+'/pickles/num_moves'+str(pid),'wb'))
    #total = pickle.load(open(os.getcwd()+'/pickles/num_moves6503000','rb'))
    print(total['6502997'])
    # plt.hist(total['6502997'],bins=l[::10])
    # plt.hist(total['6502995'],bins=l[::10])
    # plt.hist(total['6502990'],bins=l[::10])
    # plt.hist(total['6502996'],bins=l[::10]) # fig 1
    # plt.hist(total['6502963'],bins=l[::10])
    # plt.hist(total['6502964'],bins=l[::10])
    # plt.hist(total['6502966'],bins=l[::10])
    # plt.hist(total['6502967'],bins=l[::10])
    # plt.hist(total['6502968'],bins=l[::10]) # fig 2
    # plt.savefig('/Users/rohankoodli/Desktop/Figure_2.png')

    # plt.hist(total['6502969'],bins=l[::10])
    # plt.hist(total['6502970'],bins=l[::10])
    # plt.hist(total['6502976'],bins=l[::10])
    # plt.title('Distribution of number of moves taken to solve a puzzle')
    # plt.xlabel('Number of moves')
    # plt.ylabel('Frequency')
    # plt.savefig('/Users/rohankoodli/Desktop/allsinglestate.png')
    # plt.show()

def plot_experience():
    full_problems = pd.read_csv(os.getcwd()+'/movesets/full-problems-nov2016.txt', sep=" ", header="infer", delimiter='\t')
    user_df = full_problems[['uid']]
    users = np.array(user_df,dtype=int) # list of users
    experienced_players = []
    unique, counts = np.unique(users, return_counts=True) # unique and counts are same length

    # return counts

    plt.figure(figsize=(8, 7), dpi=300)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.hist(counts,bins=range(5001)[::50])
    plt.yscale('log')
    plt.ylabel('Number of players (logarithmic scale)', fontsize=20, fontweight='bold')
    plt.xlabel('Number of total puzzle solutions', fontsize=20, fontweight='bold')
    plt.xlim(0, 5000)
    #plt.show()
    plt.savefig("~/Desktop/new2.png")

# new_ms = os.getcwd() + '/movesets/moveset6-22a.txt'
#
# moveset_dataFrame = pd.read_csv(new_ms, sep=' ', header='infer', delimiter='\t')
# #puzzles_pid = moveset_dataFrame.loc[moveset_dataFrame['pid'] == 1005956]
# for i in [8627]:
#     puzzles1 = moveset_dataFrame.loc[moveset_dataFrame['uid'] == i]
#     d = list(puzzles1['move_set'])
#     for j in d[1000:1010]:
#         try:
#             #print "thunder", 392
#             s1 = ast.literal_eval(j)
#             s2 = int(s1['num_moves'])
#             print(i)
#             print(list(puzzles1['pid'])[d.index(j)])
#             print(s2)
#         #print '\n'
#         except ValueError:
#             continue
#     #print d


def num_puzzles():
    pidlist = list(ms_df['pid'])
    individual_pids = list(set(pidlist))
    print(len(individual_pids))

    over = 0
    midhigh = 0
    midlow = 0
    less = 0
    low = 0
    verylow = 0

    for i in individual_pids:
        counts = pidlist.count(i)
        if counts >= 5000: # 4
            over += 1
        elif counts < 5000 and counts >= 3000: # 19
            midhigh += 1
        elif counts < 3000 and counts >= 1000: # 35
            midlow += 1
        elif counts < 1000 and counts >= 500: # 26
            less += 1
        elif counts < 500 and counts >= 100: # 165
            low += 1
        elif counts < 100: # 8952
            verylow += 1

    print(over)
    print(midhigh)
    print(midlow)
    print(less)
    print(low)
    print(verylow)

if __name__ == '__main__':
    plot_experience()
