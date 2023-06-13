# coding:utf-8
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


import torch
import torch_npu
import torch.nn as nn
from models import EncoderDecoder
from data_utils import DataOrderScaner
import os, h5py
import constants

def evaluate(src, model, max_length):
    """
    evaluate one source sequence
    """
    m0, m1 = model
    length = len(src)
    src = torch.LongTensor(src)
    ## (seq_len, batch)
    src = src.view(-1, 1)
    length = torch.LongTensor([[length]])

    encoder_hn, H = m0.encoder(src, length)
    h = m0.encoder_hn2decoder_h0(encoder_hn)
    ## running the decoder step by step with BOS as input
    input = torch.LongTensor([[constants.BOS]])
    trg = []
    for _ in range(max_length):
        ## `h` is updated for next iteration
        o, h = m0.decoder(input, h, H)
        o = o.view(-1, o.size(2)) ## => (1, hidden_size)
        o = m1(o) ## => (1, vocab_size)
        ## the most likely word
        _, word_id = o.data.topk(1)
        word_id = word_id[0][0]
        if word_id == constants.EOS:
            break
        trg.append(word_id)
        ## update `input` for next iteration
        input = torch.LongTensor([[word_id]])
    return trg


def evaluator(args):
    """
    do evaluation interactively
    """
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                       nn.LogSoftmax())
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        while True:
            try:
                print("> ", end="")
                src = input()
                src = [int(x) for x in src.split()]
                trg = evaluate(src, (m0, m1), args.max_length)
                print(" ".join(map(str, trg)))
            except KeyboardInterrupt:
                break
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

def t2vec(args):
    torch.npu.set_compile_mode(jit_compile=False)
    "read source sequences from trj.t and write the tensor into file trj.h5"
    print("eval.......")
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        # if torch.cuda.is_available():
        #     m0.cuda()
        if torch.npu.is_available():
            m0.npu()
        m0.eval()
        vecs = []
        scaner = DataOrderScaner(os.path.join(args.data, "{}-trj.t".format(args.prefix)), args.t2vec_batch)
        scaner.load()
        i = 0
        while True:
            if i % 100 == 0:
                print("{}: Encoding {} trjs...".format(i, args.t2vec_batch))
            i = i + 1
            src, lengths, invp = scaner.getbatch()
            if src is None: break
            # if torch.cuda.is_available():
            #     src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            if torch.npu.is_available():
                src, lengths, invp = src.npu(), lengths.npu(), invp.npu()
            h, _ = m0.encoder(src, lengths)
            ## (num_layers, batch, hidden_size * num_directions)
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions)
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        ## (num_seqs, num_layers, hidden_size * num_directions)
        vecs = torch.cat(vecs)
        ## (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()
        path = os.path.join(args.data, "{}-trj.h5".format(args.prefix))
        print("=> saving vectors into {}".format(path))
        with h5py.File(path, "w") as f:
            for i in range(m0.num_layers):
                f["layer"+str(i+1)] = vecs[i].squeeze(0).numpy()
        #torch.save(vecs.data, path)
        #return vecs.data
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

#args = FakeArgs()
#args.t2vec_batch = 128
#args.num_layers = 2
#args.hidden_size = 64
#vecs = t2vec(args)
#vecs
