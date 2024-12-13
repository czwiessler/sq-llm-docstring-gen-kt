''' This script handles local interactive inference '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import spacy

from seq2seq.Models import Seq2Seq
from seq2seq.Translator import Translator
from seq2seq.Beam import Beam
from seq2seq import Constants


class Interactive(Translator):
    def __init__(self, opt):
        super().__init__(opt)

    def translate_batch(self, src_seq, src_pos):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            #- Active sentences are collected so the decoder will not run on completed sentences
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
                dec_output, *_ = self.model.decoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = self.model.tgt_word_prj(dec_output)
                word_prob[:, Constants.UNK] = -float('inf')
                word_prob = F.log_softmax(word_prob, dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #- Zero out hidden state to batch size 1
            self.model.session.zero_lstm_state(1, self.device)

            #- Encode
            src_enc, *_ = self.model.encoder(src_seq, src_pos)
            src_enc, *_ = self.model.session(src_enc)

            #- Repeat data for beam search
            n_bm = self.opt.beam_size
            n_inst, len_s, d_h = src_enc.size()
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            #- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            #- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #- Decode
            for len_dec_seq in range(1, self.model_opt.max_subseq_len + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

            hyp, scores = collect_hypothesis_and_scores(inst_dec_beams, self.opt.n_best)

        return hyp, scores
    
def interactive(opt):
    def prepare_seq(seq, max_seq_len, word2idx, device):
        ''' Prepares sequence for inference '''
        seq = nlp(seq)
        seq = [token.text for token in seq[:max_seq_len]]
        seq = [word2idx.get(w.lower(), Constants.UNK) for w in seq]
        seq = [Constants.BOS] + seq + [Constants.EOS]
        seq = np.array(seq + [Constants.PAD] * (max_seq_len - len(seq)))
        pos = np.array([pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(seq)])

        seq = torch.LongTensor(seq).unsqueeze(0)
        pos = torch.LongTensor(pos).unsqueeze(0)
        return seq.to(device), pos.to(device)

    #- Load preprocessing file for vocabulary
    prepro = torch.load(opt.prepro_file)
    src_word2idx = prepro['dict']['src']
    tgt_idx2word = {idx: word for word, idx in prepro['dict']['tgt'].items()}
    del prepro # to save memory

    #- Prepare interactive shell
    nlp = spacy.blank('en')
    s2s = Interactive(opt)
    max_seq_len = s2s.model_opt.max_subseq_len
    print('[Info] Model opts: {}'.format(s2s.model_opt))

    #- Interact with console
    console_input = ''
    console_output = '[Seq2Seq](score:--.--) human , what do you have to say ( type \' exit \' to quit ) ?\n[Human] '
    while True:
        console_input = input(console_output) # get user input
        if console_input == 'exit':
            break
        seq, pos = prepare_seq(console_input, max_seq_len, src_word2idx, s2s.device)
        console_output, score = s2s.translate_batch(seq, pos)
        console_output = console_output[0][0]
        score = score[0][0]
        console_output = '[Seq2Seq](score:{score:2.2f}) '.format(score=score.item()) + \
            ' '.join([tgt_idx2word.get(word, Constants.UNK_WORD) for word in console_output]) + '\n[Human] '
    
    print('[Seq2Seq](score:--.--) thanks for talking with me !')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True, help='Path to model .chkpt file')
    parser.add_argument('-prepro_file', required=True, help='Path to preprocessed data for vocab')
    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.n_best = 1

    interactive(opt)
