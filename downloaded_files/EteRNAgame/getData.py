# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 18:30:26 2017

@author: Rohan
"""

#from selenium import webdriver
#from selenium.webdriver.common.keys import Keys
#import time
import RNA
import pandas as pd
import os


def getPid():
    '''
    Gets the puzzle IDs for all single-state RNA puzzles

    :return: Puzzle IDs of all single-state puzzles
    '''
    ps = pd.read_csv(os.getcwd()+'/movesets/puzzle-structure-data.txt',sep=' ',header='infer',delimiter='\t')
    ps = ps.dropna(subset=['constraints'])
    c = list(ps['constraints'])
    p = list(ps['pid'])
    bad = []
    for i in range(len(p)):
        if 'SHAPE,0' not in c[i] or 'SHAPE,1' in c[i] or 'SOFT' in c[i]:
            bad.append(p[i])
    single = [x for x in p if x not in bad]

    return single


def getData_pid(pid,pidList,movesets,structure): # returns moveset and puzzzle structure together
  i1 = pidList.index(pid)
  #return movesets[num]
  pid_structure = structure['pid']
  pid_puzzleList = list(pid_structure)
  i2 = pid_puzzleList.index(pid)

  return movesets[i1], structure['structure'][i2]


def getStructure(sequence):
    '''
    Uses RNAfold to calculate structures and energies

    :param sequence: An RNA sequence
    :return: The encoded structure and the Gibbs free energy in kcal/mol
    '''
    base_seq = []
    for i in sequence:
        if i == 1:
            base_seq.append('A')
        elif i == 2:
            base_seq.append('U')
        elif i == 3:
            base_seq.append('G')
        elif i == 4:
            base_seq.append('C')

    base_seq = ''.join(base_seq)
    struc,energy = RNA.fold(base_seq)
    e = [energy]+((len(base_seq)-1)*[0.0])
    return struc,e


def getTargetEnergy(sequence,structure):
    '''
    Uses RNAfold to calculate energies of RNA sequences folded into target structures

    :param sequence: An RNA sequence
    :param structure: A target structure in dot-bracket notation
    :return: Gibbs free energy in kcal/mol
    '''

    base_seq = []
    for i in sequence:
        if i == 1:
            base_seq.append('A')
        elif i == 2:
            base_seq.append('U')
        elif i == 3:
            base_seq.append('G')
        elif i == 4:
            base_seq.append('C')
    base_seq = ''.join(base_seq)

    struc = []
    for j in structure:
        if j == 1:
            struc.append('.')
        elif j == 2:
            struc.append('(')
        elif j == 3:
            struc.append(')')
    struc = ''.join(struc)

    target_energy = RNA.energy_of_structure(base_seq,struc,0)
    e = [target_energy]+((len(base_seq)-1)*[0.0])

    return e

'''
def getStructure_0(sequence): # gets structure using Vienna algorithm # DEPRECATED
    driver = webdriver.Chrome('/Users/rohankoodli/Desktop/chromedriver')
    driver.get("http://rna.tbi.univie.ac.at//cgi-bin/RNAWebSuite/RNAfold.cgi")
    inputElement = driver.find_element_by_id("SCREEN")
    for i in sequence:
        if i == 1:
            inputElement.send_keys('A')
        elif i == 2:
            inputElement.send_keys('U')
        elif i == 3:
            inputElement.send_keys('G')
        elif i == 4:
            inputElement.send_keys('C')

    driver.find_element_by_class_name('proceed').click()
    time.sleep(20)
    web_struc = driver.find_element_by_id('MFE_structure_span').text

    struc = []
    for i in web_struc:
        if i == '.' or i == '(' or i == ')':
            struc.append(i)
    struc = ''.join(struc)
    return struc


#driver = webdriver.Chrome('/Users/rohankoodli/Desktop/chromedriver')
#driver.get("http://rna.tbi.univie.ac.at//cgi-bin/RNAWebSuite/RNAfold.cgi")
#inputElement = driver.find_element_by_id("SCREEN")
#seq = ['G','G','G','A','A','A','C','C','C']
#for i in seq:
#    inputElement.send_keys(i)
#driver.find_element_by_class_name('proceed').click()
#
#time.sleep(20)
#web_struc = driver.find_element_by_id('MFE_structure_span').text
#
#print web_struc
#
#
#struc = []
#for i in (web_struc):
#    if i == '.' or i == '(' or i == ')':
#        struc.append(i)
#
#struc = ''.join(struc)
#print struc
#print type(struc)
'''
