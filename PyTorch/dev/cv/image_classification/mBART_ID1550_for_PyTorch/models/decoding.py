#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""
pytorch-dl
Created by raj at 22:21 
Date: February 18, 2020	
"""
import math

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from dataset.iwslt_data import subsequent_mask
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def greedy_decode(model, src_tokens, src_mask, start_symbol, max=100):
    memory = model.encoder(src_tokens, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src_tokens.data)
    for i in range(max):
        out = model.decoder(Variable(ys), memory, src_mask,
                            Variable(subsequent_mask(ys.size(1)).type_as(src_tokens.data)))
        prob, logit = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src_tokens.data).fill_(next_word)], dim=1)
    return ys


def beam_search(model, src_tokens, src_mask, start_symbol, pad_symbol, max=100):
    # This is forcing the model to match the source length
    beam_size=5
    topk = [[[], .0, None]]  # [sequence, score, key_states]

    memory = model.encoder(src_tokens, src_mask)
    input_tokens = torch.ones(1, 1).fill_(start_symbol).type_as(src_tokens.data)

    for _ in range(max):
        candidates = []
        for i, (seq, score, key_states) in enumerate(topk):
            # get decoder output
            if seq:
                # convert list of tensors to tensor list and add a new dimension for batch
                input_tokens = torch.stack(seq).unsqueeze(0)

            # get decoder output
            out = model.decoder(Variable(input_tokens), memory, src_mask,
                                Variable(subsequent_mask(input_tokens.size(1)).type_as(src_tokens.data)))
            states = out[:, -1]

            lprobs, logit = model.generator(states)
            lprobs[:, pad_symbol] = -math.inf  # never select pad

            # Restrict number of candidates to only twice that of beam size
            prob, indices = torch.topk(lprobs, 2 * beam_size, dim=1, largest=True, sorted=True)

            # calculate scores
            for (idx, val) in zip(indices[0], prob[0]):
                candidate = [seq + [torch.tensor(idx).to(f'npu:{NPU_CALCULATE_DEVICE}')], score + val.item(), i]
                candidates.append(candidate)

            # order all candidates by score, select k-best
            topk = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    best_hypothesis = [idx.item() for idx in topk[0][0]]
    best_hypothesis_tensor = torch.tensor(best_hypothesis).unsqueeze(0)
    return best_hypothesis_tensor


def batched_beam_search(model, src_tokens, src_mask, start_symbol, pad_symbol, max=100):
    # This is forcing the model to match the source length
    beam_size=5
    topk = [[[], .0, None]]  # [sequence, score, key_states]
    bs, tokens = src_tokens.size()

    memory = model.encoder(src_tokens, src_mask)
    input_tokens = torch.ones(1, 1).fill_(start_symbol).type_as(src_tokens.data)

    for _ in range(max):
        candidates = []
        # Prepare all the hypothesises to be decoded in a single batch
        batch_seq = [torch.tensor(seq).to(f'npu:{NPU_CALCULATE_DEVICE}') for i, (seq, score, key_states) in enumerate(topk)]

        # get decoder output
        if batch_seq[0].size(0) > 0:
            # convert list of tensors to tensor list and add a new dimension for batch
            input_tokens = torch.stack(batch_seq)

        # get decoder output
        out = model.decoder(Variable(input_tokens), memory, src_mask,
                            Variable(subsequent_mask(input_tokens.size(1)).type_as(src_tokens.data)))
        states = out[:, -1]
        lprobs, logit = model.generator(states)
        lprobs[:, pad_symbol] = -math.inf  # never select pad

        # Restrict number of candidates to only twice that of beam size
        prob, indices = torch.topk(lprobs, 2 * beam_size, dim=1, largest=True, sorted=True)

        # calculate scores
        for i, (seq, score, key_states) in enumerate(topk):
            for (idx, val) in zip(indices[i], prob[i]):
                candidate = [seq + [torch.tensor(idx).to(f'npu:{NPU_CALCULATE_DEVICE}')], score + val.item(), i]
                candidates.append(candidate)

        # order all candidates by score, select k-best
        topk = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    best_hypothesis = [idx.item() for idx in topk[0][0]]
    best_hypothesis_tensor = torch.tensor(best_hypothesis).unsqueeze(0)
    return best_hypothesis_tensor


def batch_decode(model, src_tokens, src_mask, src_len, pad_index, sos_index, eos_index, max_len=60):
    # input batch
    bs = len(src_len)

    src_enc = model.encoder(src_tokens, src_mask)
    assert src_enc.size(0) == bs

    # generated sentences
    generated = src_len.new(max_len, bs)  # upcoming output
    generated.fill_(pad_index)  # fill upcoming ouput with <PAD>
    generated[0].fill_(sos_index)  # fill 0th index with <SOS>

    cur_len = 1
    gen_len = src_len.clone().fill_(1)
    unfinished_sents = src_len.clone().fill_(1)
    # print(unfinished_sents)

    while cur_len < max_len:
        # print(generated[:cur_len])
        # compute word scores
        tensor = model.decoder(
            tokens=Variable(generated[:cur_len].transpose(0, 1)),
            memory=src_enc,
            src_mask=src_mask,
            trg_mask=Variable(subsequent_mask(cur_len).type_as(src_tokens.data)),
        )
        # print(tensor.shape)
        tensor = tensor.data[:, -1].type_as(src_enc)  # (bs, dim)
        # print(tensor.shape)
        prob, logit = model.generator(tensor)
        # print(prob.shape, logit.shape)
        # x, next_word = torch.max(prob, dim=1)
        next_words = torch.topk(prob, 1)[1].squeeze(1)
        # print(next_words)
        # print(next_words, generated[:cur_len])

        # update generations / lengths / finished sentences / current length
        # print(next_words * unfinished_sents)
        generated[cur_len] = next_words * unfinished_sents + pad_index * (1 - unfinished_sents)
        gen_len.add_(unfinished_sents)
        unfinished_sents.mul_(next_words.ne(eos_index).long())
        cur_len = cur_len + 1

        # break
        # assert tensor.size() == (1, bs, self.dim), (cur_len, max_len,
        #                                             src_enc.size(), tensor.size(), (1, bs, self.dim))
        # tensor = tensor.data[-1, :, :].type_as(src_enc)  # (bs, dim)
        # scores = self.pred_layer.get_scores(tensor)      # (bs, n_words)
    return generated.transpose(0, 1)


def beam_decode(model, src_tokens, src_mask, src_len, pad_index, sos_index, eos_index, max_len=60, n_words=None):
    # input batch
    bs = len(src_len)

    bs = 5
    src_enc = model.encoder(src_tokens, src_mask)
    # print('enc-size', src_enc.shape)
    # print('src-mask', src_mask.shape)
    src_enc = src_enc.squeeze()
    src_mask = src_mask.squeeze()
    # print(src_enc.unsqueeze(0).shape[0:])
    # print(src_mask.shape)
    src_mask = src_mask.unsqueeze(0).expand((bs,) + src_mask.shape).contiguous().view((bs, 1) + src_mask.shape)
    src_enc = src_enc.unsqueeze(0).expand((bs,) + src_enc.shape).contiguous().view((bs,) + src_enc.shape)
    # beam_size=1
    # src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view(
    #     (bs * beam_size,) + src_enc.shape[1:])

    # enc-size: torch.Size([5, 57, 512])
    # src-mask: torch.Size([5, 1, 57])
    # print('enc-size', src_enc.shape)
    # print('src-mask', src_mask.shape)
    assert src_enc.size(0) == bs

    # generated sentences
    generated = src_len.new(max_len, bs)  # upcoming output
    generated.fill_(pad_index)  # fill upcoming ouput with <PAD>
    generated[0].fill_(sos_index)  # fill 0th index with <SOS>

    cur_len = 1
    gen_len = src_len.clone().fill_(1)
    unfinished_sents = src_len.clone().fill_(1)
    # print(unfinished_sents)

    # scores for each sentence in the beam
    beam_scores = src_enc.new(bs,).fill_(0)
    beam_scores[0] = -1e9
    beam_scores = beam_scores.view(-1)
    print(beam_scores)

    while cur_len < max_len:
        # print(generated[:cur_len].transpose(0, 1))
        # compute word scores
        trg_mask = subsequent_mask(cur_len).type_as(src_tokens.data)
        # print('trg mask', trg_mask.shape)
        tensor = model.decoder(
            tokens=Variable(generated[:cur_len].transpose(0, 1)),
            memory=src_enc,
            src_mask=src_mask,
            trg_mask=Variable(trg_mask),
        )
        print(tensor.shape)
        tensor = tensor.data[:, -1].type_as(src_enc)  # (bs, dim)
        print(tensor.shape)
        prob, logit = model.generator(tensor)
        print(prob.shape, logit.shape)
        # x, next_word = torch.max(prob, dim=1)
        next_words = torch.topk(prob, 1)[1].squeeze(1)
        next_scores, next_words = torch.topk(prob, 2 * bs, dim=1, largest=True, sorted=True)
        print(next_words.shape, next_scores.shape)

        next_batch_beam = []
        # next sentence beam content
        next_sent_beam = []
        # next words for this sentence
        for idx, value in zip(next_words, next_scores):
            # get beam and word IDs
            beam_id = idx // n_words
            word_id = idx % n_words
            next_sent_beam.append((value, word_id,  bs + beam_id))
        # next_batch_beam.extend(next_sent_beam)
        # print(next_sent_beam)
        for x in next_sent_beam:
            print(x[0])
        beam_scores = beam_scores.new([x[0] for x in next_sent_beam])
        beam_words = generated.new(next_sent_beam[1])
        beam_idx = src_len.new(next_sent_beam[2])
        exit(0)

        # update generations / lengths / finished sentences / current length
        # print(next_words * unfinished_sents)
        generated[cur_len] = next_words * unfinished_sents + pad_index * (1 - unfinished_sents)
        gen_len.add_(unfinished_sents)
        # unfinished_sents.mul_(next_words.ne(eos_index).long())
        cur_len = cur_len + 1
        print(generated)

        # break
        # assert tensor.size() == (1, bs, self.dim), (cur_len, max_len,
        #                                             src_enc.size(), tensor.size(), (1, bs, self.dim))
        # tensor = tensor.data[-1, :, :].type_as(src_enc)  # (bs, dim)
        # scores = self.pred_layer.get_scores(tensor)      # (bs, n_words)
    return generated.transpose(0, 1)


def generate_beam(model, src, src_mask, src_len,
                  pad_index,
                  sos_index,
                  eos_index,
                  emb_dim,
                  vocab_size,
                  beam_size=5,
                  length_penalty=False,
                  early_stopping=False,
                  max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # check inputs
        src_enc = model.encoder(src, src_mask)
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        print(bs)
        n_words = vocab_size

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(pad_index)                   # fill upcoming ouput with <PAD>
        generated[0].fill_(sos_index)                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache = {'slen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        def make_std_mask(tgt, pad):
            "Create a mask to hide padding and future words."
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
            return tgt_mask

        while cur_len < max_len:
            print(generated[:cur_len].transpose(0, 1).shape)
            # compute word scores
            tensor = model.decoder(
                tokens=Variable(generated[:cur_len].transpose(0, 1)),
                memory=src_enc,
                src_mask=src_mask,
                trg_mask=Variable(make_std_mask(generated[:cur_len].transpose(0, 1), pad_index).type_as(src.data)),
            )
            print('before', tensor.shape)
            tensor = tensor.view(-1, bs * beam_size, emb_dim)
            print('after', tensor.shape)

            assert tensor.size() == (1, bs * beam_size, emb_dim)
            tensor = tensor.data[-1, :, :]               # (bs * beam_size, dim)
            scores, logit = model.generator(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(logit, dim=-1)  # (bs * beam_size, n_words)
            print(scores.shape)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = eos_index

        # sanity check
        assert (decoded == eos_index).sum() == 2 * bs

        return decoded, tgt_len


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty
