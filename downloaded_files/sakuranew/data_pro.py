import pickle

f = open('./data/birth_date_train.txt', 'r', encoding='utf-8')
train_x = []
train_y = []
test_x = []
test_y = []
while True:
    content = f.readline()
    if content == '':
        break

    content = content.strip().split('#')
    # get entity name
    en1 = content[0]
    en2 = content[1]

    relation = int(content[2])
    train_y.append(relation)
    # put the same entity pair sentences into a dict
    tup = (en1, en2)
    label_tag = 0
    # if tup not in train_sen:
    #     train_sen[tup] = []
    #     train_sen[tup].append([])
    #     y_id = relation
    #     label_tag = 0
    #     label = [0 for i in range(len(relation2id))]
    #     label[y_id] = 1
    #     train_ans[tup] = []
    #     train_ans[tup].append(label)
    # else:
    #     y_id = relation
    #     label_tag = 0
    #     label = [0 for i in range(len(relation2id))]
    #     label[y_id] = 1

    #     temp = find_index(label, train_ans[tup])
    #     if temp == -1:
    #         train_ans[tup].append(label)
    #         label_tag = len(train_ans[tup]) - 1
    #         train_sen[tup].append([])
    #     else:
    #         label_tag = temp

    sentence = content[3]

    en1pos = 0
    en2pos = 0

    # For Chinese
    en1pos = sentence.find(en1)
    if en1pos == -1:
        en1pos = 0
    en2pos = sentence.find(en2)
    if en2pos == -1:
        en2pos = 0

    output = []

    # #Embeding the position
    # for i in range(fixlen):
    #     word = word2id['BLANK']
    #     rel_e1 = pos_embed(i - en1pos)
    #     rel_e2 = pos_embed(i - en2pos)
    #     output.append([word, rel_e1, rel_e2])
    s = sentence
    # s=[s[i] for i in range(len(s))]
    train_x.append(s)

print('reading test data ...')

f = open('./data/birth_date_test.txt', 'r', encoding='utf-8')

while True:
    content = f.readline()
    if content == '':
        break

    content = content.strip().split('#')
    # get entity name
    en1 = content[0]
    en2 = content[1]

    relation = int(content[2])
    test_y.append(relation)
    # put the same entity pair sentences into a dict
    tup = (en1, en2)
    label_tag = 0
    # if tup not in train_sen:
    #     train_sen[tup] = []
    #     train_sen[tup].append([])
    #     y_id = relation
    #     label_tag = 0
    #     label = [0 for i in range(len(relation2id))]
    #     label[y_id] = 1
    #     train_ans[tup] = []
    #     train_ans[tup].append(label)
    # else:
    #     y_id = relation
    #     label_tag = 0
    #     label = [0 for i in range(len(relation2id))]
    #     label[y_id] = 1

    #     temp = find_index(label, train_ans[tup])
    #     if temp == -1:
    #         train_ans[tup].append(label)
    #         label_tag = len(train_ans[tup]) - 1
    #         train_sen[tup].append([])
    #     else:
    #         label_tag = temp

    sentence = content[3]

    en1pos = 0
    en2pos = 0

    # For Chinese
    en1pos = sentence.find(en1)
    if en1pos == -1:
        en1pos = 0
    en2pos = sentence.find(en2)
    if en2pos == -1:
        en2pos = 0

    output = []

    # #Embeding the position
    # for i in range(fixlen):
    #     word = word2id['BLANK']
    #     rel_e1 = pos_embed(i - en1pos)
    #     rel_e2 = pos_embed(i - en2pos)
    #     output.append([word, rel_e1, rel_e2])
    s = sentence
    # s=[s[i] for i in range(len(s))]
    test_x.append(s)
train_x = train_x[:6000]
test_x = test_x[:3000]
train_y = train_y[:6000]
test_y = test_y[:3000]
all_x = []
for target_list in train_x:
    all_x.append(target_list)
for target_list in test_x:
    all_x.append(target_list)
with open('birth_date_trainx.txt', 'w', encoding='utf-8') as f:
    for target_list in train_x:
        f.write(target_list)
        f.write('\n')

with open('birth_date_trainy.pickle', 'wb') as f:
    pickle.dump(train_y, f)
with open('birth_date_testx.txt', 'w', encoding='utf-8') as f:
    for target_list in test_x:
        f.write(target_list)
        f.write('\n')
with open('birth_date_testy.pickle', 'wb') as f:
    pickle.dump(test_y, f)
with open('birth_date_x.txt', 'w', encoding='utf-8') as f:
    for target_list in all_x:
        f.write(target_list)
        f.write('\n')
