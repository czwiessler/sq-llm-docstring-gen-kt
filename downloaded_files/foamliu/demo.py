import random

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from config import batch_size
from data_gen import AiChallengerDataset
from data_gen import pad_collate

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval()

    dset = AiChallengerDataset()
    dset.set_mode('valid')
    valid_loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    chosen_samples = range(len(dset))
    _ids = random.sample(chosen_samples, 10)

    _pred_ids = []

    for i, data in enumerate(valid_loader):
        contexts, questions, _, alternatives = data
        contexts = Variable(contexts.long().cuda())
        questions = Variable(questions.long().cuda())
        alternatives = Variable(alternatives.long().cuda())

        preds = model.forward(contexts, questions, alternatives)
        _, pred_ids = torch.max(preds, dim=1)
        _pred_ids += list(pred_ids.cpu().numpy())

    print('len(_pred_ids): ' + str(len(_pred_ids)))

    for id in _ids:
        context = dset[id][0]
        context = [''.join([dset.QA.IVOCAB[id] for id in sentence]) for sentence in context]
        context = '。'.join(context).replace('<EOS>', '')
        question = dset[id][1]
        question = ''.join([dset.QA.IVOCAB[id] for id in question]).replace('<EOS>', '')
        answer = dset[id][3][dset[id][2]]
        answer = dset.QA.IVOCAB[answer]
        alternative = dset[id][3]
        alternative = [dset.QA.IVOCAB[id] for id in alternative]

        pred_id = _pred_ids[id]
        pred = alternative[pred_id]
        alternative = '|'.join(alternative)

        print('文章：' + context)
        print('提问：' + question)
        print('备选答案：' + alternative)
        print('标准答案：' + answer)
        print('电脑抢答：' + pred)
        print()
