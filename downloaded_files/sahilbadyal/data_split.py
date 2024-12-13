"""split data into training and validation sets"""
import csv
from models.defs import LID

with open('./data/trainingData.csv', 'r') as csvfile:
    next(csvfile) #skip headers
    data = list(csv.reader(csvfile, delimiter=','))

    #Map every language to an ID
    langs = set([language.strip() for _,language in data])
    print LID
    ID = LID

    #Write first 306 items to training set and the rest to validation set
    cnt = [0 for _ in range(len(langs))]
    with open('./data/trainEqual.csv', 'w') as train:
        with open('./data/valEqual.csv', 'w') as val:
            for line in data:
                filepath, language = map(str.strip, line)
                id_lang = ID[language]

                if (cnt[id_lang] < 306):
                    train.write(filepath[:-4] + ',' + str(id_lang) + '\n')
                else:
                    val.write(filepath[:-4] + ',' + str(id_lang) + '\n')
                cnt[id_lang] += 1
