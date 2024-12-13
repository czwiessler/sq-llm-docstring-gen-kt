# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from tensorboardX import SummaryWriter

weight_matrix = torch.tensor(np.load('./data/title_weight_matrix.npy'), dtype=torch.float32)
sequence = torch.tensor(np.load('./data/title_sequence.npy'), dtype=torch.int64)
writer = SummaryWriter(log_dir='./logs/RNNAE')


class Encoder(nn.Module):
    def __init__(self, emb_size, hidden_size, embedding):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
    
    def forward(self, x, hidden):
        output = self.embedding(x)
        output, hidden = self.gru(output, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, emb_size, dict_size, embedding):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, dict_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x, hidden):
        output = self.embedding(x)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(F.relu(self.out(output)))
        return output, hidden


class RNNAE(object):
    def __init__(self, weight_matrix, hidden_size):
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(weight_matrix.shape[0],
                                      weight_matrix.shape[1],
                                      _weight=torch.tensor(weight_matrix, dtype=torch.float32))
        self.encoder = Encoder(
            emb_size=weight_matrix.shape[1],
            hidden_size=hidden_size,
            embedding=self.embedding)
        self.decoder = Decoder(
            dict_size=weight_matrix.shape[0],
            emb_size=weight_matrix.shape[1],
            hidden_size=hidden_size,
            embedding=self.embedding)
        self.optimizer = optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=1e-3)
        self.criterion = nn.NLLLoss()
    
    def _train(self, batch_x, sentence_length):
        self.optimizer.zero_grad()
        loss = 0
        _, decoder_hidden = self.encoder(batch_x, hidden=None)
        decoder_input = torch.ones(batch_x.shape[0], 1, dtype=torch.int64)
        for i in range(1, sentence_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1, dim=-1)
            decoder_input = topi.squeeze(-1).detach()
            loss += self.criterion(decoder_output.squeeze(1), batch_x[:, i])
        loss.backward()
        self.optimizer.step()
        return loss.item() / sentence_length
    
    def train(self, X, batch_size=64, epoch=10):
        global_step = 0
        for e in range(epoch):
            pointer = 0
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            while pointer < X.shape[0]:
                batch_x = X_shuffled[pointer:(pointer + batch_size)]
                max_sentence_length = (batch_x != 0).sum(dim=-1).max()
                mean_loss = self._train(batch_x, sentence_length=int(max_sentence_length))
                print(mean_loss, 'batch%:', round((pointer / X.shape[0]) * 100, 4), 'epoch:', e)
                pointer += batch_size
                writer.add_scalar(tag='loss', scalar_value=mean_loss, global_step=global_step)
                global_step += 1
            self.save_model()
    
    def encode(self, X, batch_size=64):
        pointer = 0
        results_=[]
        # results_ = np.zeros((1, self.hidden_size))
        while pointer < X.shape[0]:
            batch_x = X[pointer:(pointer + batch_size)]
            max_sentence_length = (batch_x != 0).sum(dim=-1).max()
            out, hidden = self.encoder(batch_x[:, :max_sentence_length], hidden=None)
            results_.append(hidden.squeeze(0).detach().numpy())
            pointer += batch_size
        return np.concatenate(results_)
    
    def save_model(self, model_path='./RNN_AE'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        print("saving models")
        torch.save(self.encoder, model_path + '/encoder.pkl')
        torch.save(self.decoder, model_path + '/decoder.pkl')
    
    def load_model(self, model_path='./RNN_AE'):
        print("loading models")
        self.encoder = torch.load(model_path + '/encoder.pkl')
        self.decoder = torch.load(model_path + '/decoder.pkl')


if __name__ == '__main__':
    rnnae = RNNAE(weight_matrix=weight_matrix, hidden_size=50)
    rnnae.train(X=sequence, batch_size=128, epoch=2)
    rnnae.save_model()
    results = rnnae.encode(sequence, batch_size=128)
    del sequence
    del weight_matrix
    np.save('./output/RNNAE/encoded_title', results)
