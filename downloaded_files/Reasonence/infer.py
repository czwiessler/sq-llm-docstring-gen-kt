import json
from metrics import metrics, traitPairs, traitTriplets, offsetTraitTriplets, offsetTraitPairs
import random

training_file = open('./training.json', 'r')
training = json.load(training_file)


def randomWeight():
    return random.uniform(1.0, 1.2)


def p(sex, metric):
    if metric not in training[sex]:
        return 0

    p_metricsex = (training[sex][metric])/training[sex + "Count"]
    p_sex = 0.5

    m_metric = 0
    f_metric = 0

    if metric in training['female']:
        f_metric = training["female"][metric]
    
    if metric in training['male']:
        m_metric = training["male"][metric]
    

    p_metric = (m_metric + f_metric)/(training["femaleCount"]+training["maleCount"])

    return ((p_metricsex*p_sex)/p_metric)
    

def infer(name: str):
    name = name.lower()
    c_name = ''

    for c in name:
        if ord(c) >= ord('a') and ord(c) <= ord('z'):
            c_name = c_name + c

    name = c_name

    mets = metrics(name)
    male = 0
    female = 0

    for i in mets:
        weight = 1
        num = i[0]

        if num in ['0','1','2','3']:
            weight = weight * (4 - int(num))**2


        female = female + (p("female", i) * weight * randomWeight())
        male = male + (p("male", i) * weight * randomWeight())

    mets = []
    mets = mets + traitPairs(name)
    mets = mets + traitTriplets(name)
    mets = mets + offsetTraitTriplets(name)
    mets = mets + offsetTraitPairs(name)

    for met_set in mets:
        mm = 0
        ff = 0

        for met in met_set:
            weight = 1
            num = met[0]

            if num in ['0','1','2','3']:
                weight = weight * (4 - int(num))**2


            ff = ff + p("female", met) * weight * randomWeight()
            mm = mm + p("male", met) * weight * randomWeight()

        if len(met_set) < 1:
            continue

        mm = mm/len(met_set)
        ff = ff/len(met_set)

        male = male + (mm)
        female = female + (ff)

    diff = (abs(female - male)/((female + male)/2))*100

    inference = ''

    if diff >= 15:
        if female > male:
            inference = 'f'
        else:
            inference = 'm'
    else:
        inference = 'u'

    return (male, female, inference)