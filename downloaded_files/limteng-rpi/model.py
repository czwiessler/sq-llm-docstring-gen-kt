import torch
import torch.nn as nn
import torch.nn.utils.rnn as R

import constant as C
from util import load_embedding_from_file, build_signal_embed
from module import Linear, LSTM, CRF, Linears, CharCNN, CharCNNFF


class LstmCnn(nn.Module):

    def __init__(self,
                 vocabs,
                 word_embed_file, word_embed_dim,
                 char_embed_dim, char_filters, char_feat_dim,
                 lstm_hidden_size,
                 lstm_dropout=0, feat_dropout=0,
                 parameters=None
                 ):
        super(LstmCnn, self).__init__()

        self.vocabs = vocabs
        self.label_size = len(self.vocabs['label'])
        # input features
        if parameters is not None:
            self.word_embed = nn.Embedding(parameters['word_embed_num'],
                                           parameters['word_embed_dim'],
                                           padding_idx=C.PAD_INDEX)
        else:
            self.word_embed = load_embedding_from_file(word_embed_file,
                                                       word_embed_dim,
                                                       vocabs['token'],
                                                       vocabs['embed'],
                                                       vocabs['form'],
                                                       padding_idx=C.PAD_INDEX,
                                                       trainable=True)
        self.char_embed = CharCNNFF(len(vocabs['char']),
                                    char_embed_dim,
                                    char_filters,
                                    output_size=char_feat_dim)
        self.word_dim = self.word_embed.embedding_dim
        self.char_dim = self.char_embed.output_size
        self.feat_dim = self.char_dim + self.word_dim
        # layers
        self.lstm = LSTM(input_size=self.feat_dim,
                         hidden_size=lstm_hidden_size,
                         batch_first=True,
                         bidirectional=True)
        self.output_linear = Linear(self.lstm.output_size,
                                    self.label_size)
        self.crf = CRF(vocabs['label'])
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)

    @property
    def params(self):
        return {
            'word_embed_num': self.word_embed.num_embeddings,
            'word_embed_dim': self.word_embed.embedding_dim
        }

    def forward_nn(self, token_ids, char_ids, lens):
        batch_size, seq_len = token_ids.size()
        # word representation
        word_in = self.word_embed(token_ids)
        char_in = self.char_embed(char_ids)
        char_in = char_in.view(batch_size, seq_len, self.char_dim)
        feats = torch.cat([word_in, char_in], dim=2)
        feats = self.feat_dropout(feats)

        # LSTM layer
        lstm_in = R.pack_padded_sequence(feats, lens.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)

        # output linear layer
        linear_out = self.output_linear(lstm_out)
        return linear_out

    def forward(self, token_ids, char_ids, lens, labels):
        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score

        return loglik, logits

    def predict(self, token_ids, char_ids, lens):
        self.eval()
        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        _scores, preds = self.crf.viterbi_decode(logits, lens)
        preds = preds.data.tolist()
        self.train()
        return preds


class LstmCnnDfc(nn.Module):
    def __init__(self,
                 vocabs,
                 counters,
                 word_embed_file, word_embed_dim,
                 char_embed_dim, char_filters, char_feat_dim,
                 lstm_hidden_size,
                 lstm_dropout=0.5, feat_dropout=0.5, signal_dropout=0,
                 ctx_size=5,
                 use_signal=True,
                 parameters=None,
                 ):
        assert char_feat_dim >= word_embed_dim
        super(LstmCnnDfc, self).__init__()

        self.vocabs = vocabs
        self.label_size = len(self.vocabs['label'])
        self.use_signal = use_signal

        # input features
        if parameters is not None:
            self.word_embed = nn.Embedding(parameters['word_embed_num'],
                                           parameters['word_embed_dim'])
        else:
            self.word_embed = load_embedding_from_file(word_embed_file,
                                                       word_embed_dim,
                                                       vocabs['token'],
                                                       vocabs['embed'],
                                                       vocabs['form'],
                                                       padding_idx=C.PAD_INDEX,
                                                       trainable=True)
        self.char_embed = CharCNNFF(len(vocabs['char']),
                                    char_embed_dim,
                                    char_filters,
                                    output_size=char_feat_dim)
        if use_signal:
            if parameters is not None:
                self.signal_embed = nn.Embedding(parameters['signal_embed_num'],
                                                 parameters['signal_embed_dim'])
            else:
                self.signal_embed = build_signal_embed(counters['embed'],
                                                       counters['token'],
                                                       vocabs['token'],
                                                       vocabs['form'])
        self.word_dim = self.word_embed.embedding_dim
        self.char_dim = self.char_embed.output_size
        self.feat_dim = self.char_dim
        self.signal_dim = self.signal_embed.embedding_dim
        self.ctx_size = ctx_size
        # layers
        self.lstm = LSTM(input_size=self.feat_dim,
                         hidden_size=lstm_hidden_size,
                         batch_first=True,
                         bidirectional=True)
        self.output_linear = Linear(self.lstm.output_size,
                                    self.label_size)
        self.crf = CRF(vocabs['label'])
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)
        self.signal_dropout = nn.Dropout(p=signal_dropout)
        self.lstm_size = self.lstm.output_size
        self.uni_lstm_size = self.lstm_size // 2

        # word representation level
        self.word_gates = nn.ModuleList([
            Linear(self.word_dim, self.word_dim),
            Linear(self.word_dim, self.word_dim)])
        self.char_gates = nn.ModuleList([
            Linear(self.word_dim, self.word_dim),
            Linear(self.word_dim, self.word_dim)])
        if use_signal:
            self.signal_gates = nn.ModuleList([
                Linear(self.signal_dim, self.word_dim),
                Linear(self.signal_dim, self.word_dim)])

        # feature extraction level
        # context-only feature linear layers
        self.cof_linear_fwd = Linear(self.uni_lstm_size,
                                        self.uni_lstm_size)
        self.cof_linear_bwd = Linear(self.uni_lstm_size,
                                        self.uni_lstm_size)
        # hidden states gates
        self.hs_gates = nn.ModuleList([
            Linear(self.uni_lstm_size, self.uni_lstm_size)
            for _ in range(4)])
        # context-only feature gates
        self.cof_gates = nn.ModuleList([
            Linear(self.uni_lstm_size, self.uni_lstm_size)
            for _ in range(4)])
        if use_signal:
            self.crs_gates = nn.ModuleList([
                Linear(self.signal_dim * (ctx_size + 1),
                       self.uni_lstm_size)
                for _ in range(4)])

    @property
    def params(self):
        return {
            'word_embed_num': self.word_embed.num_embeddings,
            'word_embed_dim': self.word_embed.embedding_dim,
            'signal_embed_num': self.signal_embed.num_embeddings,
            'signal_embed_dim': self.signal_embed.embedding_dim
        }

    def _repr_gate(self, word, char, signal=None, idx=0):
        gate_w = self.word_gates[idx](word)
        gate_c = self.char_gates[idx](char)
        if self.use_signal:
            gate_s = self.signal_gates[idx](self.signal_dropout(signal))
            gate = gate_w + gate_c + gate_s
        else:
            gate = gate_w + gate_c
        gate = gate.sigmoid()
        return gate

    def _feat_gate(self, hs, cof, crs=None, idx=0):
        """Calculate feature extraction level gates.
        :param hs: Hidden states.
        :param cof: Context-only features.
        :param crs: Context reliability signals.
        """
        gate_h = self.hs_gates[idx](hs)
        gate_c = self.cof_gates[idx](cof)
        if self.use_signal:
            gate_s = self.crs_gates[idx](self.signal_dropout(crs))
            gate = gate_h + gate_c + gate_s
        else:
            gate = gate_h + gate_c
        gate = gate.sigmoid()
        return gate

    def forward_nn(self, token_ids, char_ids, lens):
        batch_size, seq_len = token_ids.size()
        word_dim = self.word_dim
        char_dim = self.char_dim
        signal_dim = self.signal_dim
        use_signal = self.use_signal
        ctx_size = self.ctx_size

        # word representations
        word_in = self.word_embed(token_ids)
        char_in = self.char_embed(char_ids)
        char_in = char_in.view(batch_size, seq_len, char_dim)
        signal_in = self.signal_embed(token_ids) if use_signal else None
        # combine features
        if char_dim == word_dim:
            # without additional char features
            repr_mix_gate_1 = self._repr_gate(word_in, char_in, signal_in, 0)
            repr_mix_gate_2 = self._repr_gate(word_in, char_in, signal_in, 1)
            feats = repr_mix_gate_1 * word_in + repr_mix_gate_2 * char_in
        else:
            # with additional char features
            char_in_alt = char_in[:, :, :word_dim]
            char_in_cat = char_in[:, :, word_dim:]
            repr_mix_gate_1 = self._repr_gate(word_in, char_in_alt, signal_in, 0)
            repr_mix_gate_2 = self._repr_gate(word_in, char_in_alt, signal_in, 1)
            feats = repr_mix_gate_1 * word_in + repr_mix_gate_2 * char_in_alt
            feats = torch.cat([feats, char_in_cat], dim=2)
        feats = self.feat_dropout(feats)

        # LSTM layer
        lstm_in = R.pack_padded_sequence(feats, lens.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)

        # context reliability signals (crs)
        if use_signal:
            rs_pad = lstm_out.new_zeros([batch_size, ctx_size, signal_dim],
                                        requires_grad=False)
            signal_in_padded = torch.cat([rs_pad, signal_in, rs_pad], dim=1)
            signal_in_padded = signal_in_padded.view(batch_size, -1)
            crs = signal_in_padded.unfold(1, signal_dim * (ctx_size + 1), signal_dim)
            crs_fwd = crs[:, :-ctx_size, :]
            crs_bwd = crs[:, ctx_size:, :]
        else:
            crs_fwd = crs_bwd = None

        # context-only features (cof)
        hs_pad = lstm_out.new_zeros([batch_size, 1, self.uni_lstm_size],
                                    requires_grad=False)
        hs_fwd = lstm_out[:, :, :self.uni_lstm_size]
        hs_bwd = lstm_out[:, :, self.uni_lstm_size:]
        hs_fwd_padded = torch.cat([hs_pad, hs_fwd], dim=1)[:, :-1, :]
        hs_bwd_padded = torch.cat([hs_bwd, hs_pad], dim=1)[:, 1:, :]
        cof_fwd = self.cof_linear_fwd(hs_fwd_padded).tanh()
        cof_bwd = self.cof_linear_bwd(hs_bwd_padded).tanh()

        # feature extract level gates
        feat_mix_gate_fwd_1 = self._feat_gate(hs_fwd, cof_fwd, crs_fwd, 0)
        feat_mix_gate_fwd_2 = self._feat_gate(hs_fwd, cof_fwd, crs_fwd, 1)
        feat_mix_gate_bwd_1 = self._feat_gate(hs_bwd, cof_bwd, crs_bwd, 2)
        feat_mix_gate_bwd_2 = self._feat_gate(hs_bwd, cof_bwd, crs_bwd, 3)

        # enhanced hidden states
        hs_fwd_enh = feat_mix_gate_fwd_1 * hs_fwd + feat_mix_gate_fwd_2 * cof_fwd
        hs_bwd_enh = feat_mix_gate_bwd_1 * hs_bwd + feat_mix_gate_bwd_2 * cof_bwd
        hs_enh = torch.cat([hs_fwd_enh, hs_bwd_enh], dim=2)

        # output linear layer
        linear_out = self.output_linear(hs_enh)

        return linear_out

    def forward(self, token_ids, char_ids, lens, labels):
        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score

        return loglik, logits

    def predict(self, token_ids, char_ids, lens):
        self.eval()

        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        _scores, preds = self.crf.viterbi_decode(logits, lens)
        preds = preds.data.tolist()

        self.train()
        return preds


class LstmCnnGate(nn.Module):
    def __init__(self,
                 vocabs,
                 counters,
                 word_embed_file, word_embed_dim,
                 char_embed_dim, char_filters, char_feat_dim,
                 lstm_hidden_size,
                 lstm_dropout=0.5, feat_dropout=0.5
                 ):
        # TODO: init function for saved model
        super(LstmCnnGate, self).__init__()

        self.vocabs = vocabs
        self.label_size = len(self.vocabs['label'])

        # input features
        self.word_embed = load_embedding_from_file(word_embed_file,
                                                   word_embed_dim,
                                                   vocabs['token'],
                                                   vocabs['embed'],
                                                   vocabs['form'],
                                                   padding_idx=C.PAD_INDEX,
                                                   trainable=True)
        self.char_embed = CharCNNFF(len(vocabs['char']),
                                    char_embed_dim,
                                    char_filters,
                                    output_size=char_feat_dim)
        self.word_dim = self.word_embed.embedding_dim
        self.char_dim = self.char_embed.output_size
        self.feat_dim = self.char_dim
        # layers
        self.lstm = LSTM(input_size=self.feat_dim,
                         hidden_size=lstm_hidden_size,
                         batch_first=True,
                         bidirectional=True)
        self.output_linear = Linear(self.lstm.output_size,
                                    self.label_size)
        self.crf = CRF(vocabs['label'])
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)
        self.lstm_size = self.lstm.output_size
        self.uni_lstm_size = self.lstm_size // 2

        # word representation level
        self.word_gate = Linear(self.word_dim, self.word_dim)
        self.char_gate = Linear(self.word_dim, self.word_dim)

        # feature extraction level
        # context-only feature linear layers
        self.cof_linear_fwd = Linear(self.uni_lstm_size, self.uni_lstm_size)
        self.cof_linear_bwd = Linear(self.uni_lstm_size, self.uni_lstm_size)
        # hidden states gates
        self.hs_gates = nn.ModuleList([
            Linear(self.uni_lstm_size, self.uni_lstm_size),
            Linear(self.uni_lstm_size, self.uni_lstm_size)])
        # context-only feature gates
        self.cof_gates = nn.ModuleList([
            Linear(self.uni_lstm_size, self.uni_lstm_size),
            Linear(self.uni_lstm_size, self.uni_lstm_size)])


    def _repr_gate(self, word, char):
        gate_w = self.word_gate(word)
        gate_c = self.char_gate(char)
        gate = gate_w + gate_c
        gate = gate.sigmoid()
        return gate

    def _feat_gate(self, hs, cof, idx):
        """Calculate feature extraction level gates.
        :param hs: Hidden states.
        :param cof: Context-only features.
        """
        gate_h = self.hs_gates[idx](hs)
        gate_c = self.cof_gates[idx](cof)
        gate = gate_h + gate_c
        gate = gate.sigmoid()
        return gate

    def forward_nn(self, token_ids, char_ids, lens):
        batch_size, seq_len = token_ids.size()
        word_dim = self.word_dim
        char_dim = self.char_dim

        # word representations
        word_in = self.word_embed(token_ids)
        char_in = self.char_embed(char_ids)
        char_in = char_in.view(batch_size, seq_len, char_dim)
        # combine features
        repr_mix_gate = self._repr_gate(word_in, char_in)
        feats = repr_mix_gate * word_in + (1 - repr_mix_gate) * char_in
        feats = self.feat_dropout(feats)

        # LSTM layer
        lstm_in = R.pack_padded_sequence(feats, lens.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)

        # context-only features (cof)
        hs_pad = lstm_out.new_zeros([batch_size, 1, self.uni_lstm_size],
                                    requires_grad=False)
        hs_fwd = lstm_out[:, :, :self.uni_lstm_size]
        hs_bwd = lstm_out[:, :, self.uni_lstm_size:]
        hs_fwd_padded = torch.cat([hs_pad, hs_fwd], dim=1)[:, :-1, :]
        hs_bwd_padded = torch.cat([hs_bwd, hs_pad], dim=1)[:, 1:, :]
        cof_fwd = self.cof_linear_fwd(hs_fwd_padded).tanh()
        cof_bwd = self.cof_linear_bwd(hs_bwd_padded).tanh()

        # feature extract level gates
        feat_mix_gate_fwd = self._feat_gate(hs_fwd, cof_fwd, 0)
        feat_mix_gate_bwd = self._feat_gate(hs_bwd, cof_bwd, 1)

        # enhanced hidden states
        hs_fwd_enh = feat_mix_gate_fwd * hs_fwd + (1 - feat_mix_gate_fwd) * cof_fwd
        hs_bwd_enh = feat_mix_gate_bwd * hs_bwd + (1 - feat_mix_gate_bwd) * cof_bwd
        hs_enh = torch.cat([hs_fwd_enh, hs_bwd_enh], dim=2)

        # output linear layer
        linear_out = self.output_linear(hs_enh)

        return linear_out

    def forward(self, token_ids, char_ids, lens, labels):
        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score

        return loglik, logits

    def predict(self, token_ids, char_ids, lens):
        self.eval()

        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        _scores, preds = self.crf.viterbi_decode(logits, lens)
        preds = preds.data.tolist()

        self.train()
        return preds


class LstmCnnFeatGate(nn.Module):
    def __init__(self, vocabs,
                 word_embed_file, word_embed_dim,
                 char_embed_dim, char_filters, char_feat_dim,
                 lstm_hidden_size,
                 lstm_dropout=.5, feat_dropout=.5,
                 parameters=None
                 ):
        super(LstmCnnFeatGate, self).__init__()
        assert word_embed_dim == char_feat_dim

        self.vocabs = vocabs
        self.label_size = len(self.vocabs['label'])

        # input features
        if parameters is not None:
            self.word_embed = nn.Embedding(parameters['word_embed_num'],
                                           parameters['word_embed_dim'])
        else:
            self.word_embed = load_embedding_from_file(word_embed_file,
                                                       word_embed_dim,
                                                       vocabs['token'],
                                                       vocabs['embed'],
                                                       vocabs['form'],
                                                       padding_idx=C.PAD_INDEX,
                                                       trainable=True)
        self.char_embed = CharCNNFF(len(vocabs['char']),
                                    char_embed_dim,
                                    char_filters,
                                    output_size=char_feat_dim)
        # word dim = char_dim = feat_dim in this model
        self.word_dim = self.word_embed.embedding_dim
        self.char_dim = self.char_embed.output_size
        self.feat_dim = self.word_dim
        # layers
        self.char_gate = Linear(self.char_dim, self.char_dim, bias=False)
        self.word_gate = Linear(self.word_dim, self.word_dim, bias=False)
        self.gate  = Linear(self.feat_dim, self.feat_dim, bias=False)

        self.lstm = LSTM(input_size=self.feat_dim,
                         hidden_size=lstm_hidden_size,
                         batch_first=True,
                         bidirectional=True)
        self.output_linear = Linear(self.lstm.output_size, self.label_size)
        self.crf = CRF(vocabs['label'])
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)

    @property
    def params(self):
        return {
            'word_embed_num': self.word_embed.num_embeddings,
            'word_embed_dim': self.word_embed.embedding_dim
        }

    def forward_nn(self, token_ids, char_ids, lens):
        batch_size, seq_len = token_ids.size()
        # word representations
        word_in = self.word_embed(token_ids)
        char_in = self.char_embed(char_ids)
        char_in = char_in.view(batch_size, seq_len, self.char_dim)
        gate = self.gate((self.word_gate(word_in) +
                          self.char_gate(char_in)).tanh()).sigmoid()
        feats = gate * word_in + (1 - gate) * char_in
        feats = self.feat_dropout(feats)
        # LSTM layer
        lstm_in = R.pack_padded_sequence(feats, lens.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)
        # output linear layer
        linear_out = self.output_linear(lstm_out)
        return linear_out

    def forward(self, token_ids, char_ids, lens, labels):
        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score
        return loglik, logits

    def predict(self, token_ids, char_ids, lens):
        self.eval()
        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        _scores, preds = self.crf.viterbi_decode(logits, lens)
        preds = preds.data.tolist()
        self.train()
        return preds
