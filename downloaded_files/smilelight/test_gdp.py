import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.sp import GDP

gdp_model = GDP()

train_path = '../data/gdp/train.sample.conll'
vec_path = 'D:/Data/NLP/embedding/word/sgns.zhihu.bigram-char'


gdp_model.train(train_path, dev_path=train_path, vectors_path=vec_path, save_path='./gdp_saves',
                log_dir='E:/Test/tensorboard/')

gdp_model.load('./gdp_saves')
# gdp_model.test(train_path)
word_list = ['最高', '人民', '检察院', '检察长', '张思卿']
pos_list = ['nt', 'nt', 'nt', 'n', 'nr']
heads, rels = gdp_model.predict(word_list, pos_list)
print(heads)
print(rels)