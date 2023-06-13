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


import numpy as np
import torch
import constants
from funcy import merge
from collections import namedtuple
import math


def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]

def pad_array(a, max_length, PAD=constants.PAD):
    """
    a (array[int32])
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))

def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)

def pad_arrays_pair(src, trg, keep_invp=False):
    """
    Input:
    src (list[array[int32]])
    trg (list[array[int32]])
    ---
    Output:
    src (seq_len1, batch)
    trg (seq_len2, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    TD = namedtuple('TD', ['src', 'lengths', 'trg', 'invp'])

    assert len(src) == len(trg), "source and target should have the same length"
    idx = argsort(src)
    src = list(np.array(src)[idx])
    trg = list(np.array(trg)[idx])

    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    if keep_invp == True:
        invp = torch.LongTensor(invpermute(idx))
        # (batch, seq_len) => (seq_len, batch)
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=invp)
    else:
        # (batch, seq_len) => (seq_len, batch)
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=[])

def invpermute(p):
    """
    inverse permutation
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp

def pad_arrays_keep_invp(src):
    """
    Pad arrays and return inverse permutation

    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    idx = argsort(src)
    src = list(np.array(src)[idx])
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    invp = torch.LongTensor(invpermute(idx))
    return src.t().contiguous(), lengths.view(1, -1), invp

def random_subseq(a, rate):
    """
    Dropping some points between a[3:-2] randomly according to rate.

    Input:
    a (array[int])
    rate (float)
    """
    idx = np.random.rand(len(a)) < rate
    idx[0], idx[-1] = True, True
    return a[idx]

class TDgroup:
    def __init__(self, dler, bucket, batch_size):
        self.bucket_map = {}
        self.indices = {}
        self.dler = dler
        self.batch_size = batch_size

        for i in range(bucket):
            self.bucket_map[i] = 0

        for i in range(bucket):
            self.set_iter(i)

    def __iter__(self, bucket):
        g = torch.Generator()
        g.manual_seed(self.bucket_map[bucket])

        lengths = len(self.dler.srcdata[bucket])
        indices = list(range(lengths))

        indices = sorted(indices, key=lambda x: len(self.dler.srcdata[bucket][x]))
        last_length = lengths % self.batch_size
        last_length = self.batch_size if last_length == 0 else last_length

        batches = int(math.ceil(lengths * 1. / self.batch_size))
        batches = torch.randperm(batches, ).tolist()
        first = batches[0]
        rand_indices = []
        rand_indices.append(indices[first * self.batch_size : (first + 1) * self.batch_size])
        for batch in batches[1:]:
            if batch < first:
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
            else:
                start = (batch - 1) * self.batch_size + last_length
                end = batch * self.batch_size + last_length
            rand_indices.append(indices[start:end])
        return iter(rand_indices)

    def set_iter(self, bucket):
        self.bucket_map[bucket] += 1
        self.indices[bucket] = self.__iter__(bucket=bucket)

class DataLoader():
    """
    srcfile: source file name
    trgfile: target file name
    batch: batch size
    validate: if validate = True return batch orderly otherwise return
        batch randomly
    """
    def __init__(self, srcfile, trgfile, mtafile, batch, bucketsize, validate=False):
        self.srcfile = srcfile
        self.trgfile = trgfile
        self.mtafile = mtafile

        self.batch = batch
        self.validate = validate
        #self.bucketsize = [(30, 30), (30, 50), (50, 50), (50, 70), (70, 70)]
        self.bucketsize = bucketsize

    def insert(self, s, t, m):
        for i in range(len(self.bucketsize)):
            if len(s) <= self.bucketsize[i][0] and len(t) <= self.bucketsize[i][1]:
                self.srcdata[i].append(np.array(s, dtype=np.int32))
                self.trgdata[i].append(np.array(t, dtype=np.int32))
                self.mtadata[i].append(np.array(m, dtype=np.float32))
                return 1
        return 0

    def load(self, max_num_line=0):
        self.srcdata = [[] for _ in range(len(self.bucketsize))]
        self.trgdata = [[] for _ in range(len(self.bucketsize))]
        self.mtadata = [[] for _ in range(len(self.bucketsize))]

        srcstream, trgstream, mtastream = open(self.srcfile, 'r'), open(self.trgfile, 'r'), open(self.mtafile, 'r')
        num_line = 0
        for (s, t, m) in zip(srcstream, trgstream, mtastream):
            s = [int(x) for x in s.split()]
            t = [constants.BOS] + [int(x) for x in t.split()] + [constants.EOS]
            m = [float(x) for x in m.split()]

            num_line += self.insert(s, t, m)
            if num_line >= max_num_line and max_num_line > 0: break
            if num_line % 500000 == 0:
                print("Read line {}".format(num_line))
        print("load data done")

        ## if vliadate is True we merge all buckets into one
        if self.validate == True:
            self.srcdata = np.array(merge(*self.srcdata))
            self.trgdata = np.array(merge(*self.trgdata))
            self.mtadata = np.array(merge(*self.mtadata))

            self.start = 0
            self.size = len(self.srcdata)
        else:
            self.srcdata = list(map(np.array, self.srcdata))
            self.trgdata = list(map(np.array, self.trgdata))
            self.mtadata = list(map(np.array, self.mtadata))

            self.allocation = list(map(len, self.srcdata))
            self.p = np.array(self.allocation) / sum(self.allocation)
            self.tdg = TDgroup(self, len(self.bucketsize), self.batch)
        srcstream.close(), trgstream.close(), mtastream.close()

    def getbatch_one(self):
        if self.validate == True:
            src = self.srcdata[self.start:self.start+self.batch]
            trg = self.trgdata[self.start:self.start+self.batch]
            mta = self.mtadata[self.start:self.start+self.batch]

            ## update `start` for next batch
            self.start += self.batch
            if self.start >= self.size:
                self.start = 0
            return list(src), list(trg), list(mta)
        else:
            ## select bucket
            sample = np.random.multinomial(1, self.p)
            bucket = np.nonzero(sample)[0][0]
            ## select data from the bucket
            #idx = np.random.choice(len(self.srcdata[bucket]), self.batch)
            try:
                indices = self.tdg.indices[bucket]
                idx = indices.__next__()
            except StopIteration:
                self.tdg.set_iter(bucket)
                indices = self.tdg.indices[bucket]
                idx = indices.__next__()
            src = self.srcdata[bucket][idx]
            trg = self.trgdata[bucket][idx]
            mta = self.mtadata[bucket][idx]
            return list(src), list(trg), list(mta)

    def getbatch_generative(self):
        src, trg, _ = self.getbatch_one()
        # src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
        return pad_arrays_pair(src, trg, keep_invp=False)

    def getbatch_discriminative_cross(self):
        def distance(x, y):
            return np.linalg.norm(x - y)
        a_src, a_trg, a_mta = self.getbatch_one()
        p_src, p_trg, p_mta = self.getbatch_one()
        n_src, n_trg, n_mta = self.getbatch_one()

        #p_src, p_trg, p_mta = copy.deepcopy(p_src), copy.deepcopy(p_trg), copy.deepcopy(p_mta)
        #n_src, n_trg, n_mta = copy.deepcopy(n_src), copy.deepcopy(n_trg), copy.deepcopy(n_mta)
        for i in range(len(a_src)):
            if distance(a_mta[i], p_mta[i]) > distance(a_mta[i], n_mta[i]):
                p_src[i], n_src[i] = n_src[i], p_src[i]
                p_trg[i], n_trg[i] = n_trg[i], p_trg[i]
                p_mta[i], n_mta[i] = n_mta[i], p_mta[i]

        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n

    def getbatch_discriminative_inner(self):
        """
        Test Case:
        a, p, n = dataloader.getbatch_discriminative_inner()
        i = 2
        idx_a = torch.nonzero(a[2].t()[a[3]][i])
        idx_p = torch.nonzero(p[2].t()[p[3]][i])
        idx_n = torch.nonzero(n[2].t()[n[3]][i])
        a_t = a[2].t()[a[3]][i][idx_a].view(-1).numpy()
        p_t = p[2].t()[p[3]][i][idx_p].view(-1).numpy()
        n_t = n[2].t()[n[3]][i][idx_n].view(-1).numpy()
        print(len(np.intersect1d(a_t, p_t)))
        print(len(np.intersect1d(a_t, n_t)))
        """
        a_src, a_trg = [], []
        p_src, p_trg = [], []
        n_src, n_trg = [], []

        _, trgs, _ = self.getbatch_one()
        for i in range(len(trgs)):
            trg = trgs[i][1:-1]
            if len(trg) < 10: continue
            a1, a3, a5 = 0, len(trg)//2, len(trg)
            a2, a4 = (a1 + a3)//2, (a3 + a5)//2
            rate = np.random.choice([0.5, 0.6, 0.8])
            if np.random.rand() > 0.5:
                a_src.append(random_subseq(trg[a1:a4], rate))
                a_trg.append(np.r_[constants.BOS, trg[a1:a4], constants.EOS])
                p_src.append(random_subseq(trg[a2:a5], rate))
                p_trg.append(np.r_[constants.BOS, trg[a2:a5], constants.EOS])
                n_src.append(random_subseq(trg[a3:a5], rate))
                n_trg.append(np.r_[constants.BOS, trg[a3:a5], constants.EOS])
            else:
                a_src.append(random_subseq(trg[a2:a5], rate))
                a_trg.append(np.r_[constants.BOS, trg[a2:a5], constants.EOS])
                p_src.append(random_subseq(trg[a1:a4], rate))
                p_trg.append(np.r_[constants.BOS, trg[a1:a4], constants.EOS])
                n_src.append(random_subseq(trg[a1:a3], rate))
                n_trg.append(np.r_[constants.BOS, trg[a1:a3], constants.EOS])

        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n




class DataOrderScaner():
    def __init__(self, srcfile, batch):
        self.srcfile = srcfile
        self.batch = batch
        self.srcdata = []
        self.start = 0
    def load(self, max_num_line=0):
        num_line = 0
        with open(self.srcfile, 'r') as srcstream:
            for s in srcstream:
                s = [int(x) for x in s.split()]
                self.srcdata.append(np.array(s, dtype=np.int32))
                num_line += 1
                if max_num_line > 0 and num_line >= max_num_line:
                    break
        self.size = len(self.srcdata)
        self.start = 0
    def getbatch(self):
        """
        Output:
        src (seq_len, batch)
        lengths (1, batch)
        invp (batch,): inverse permutation, src.t()[invp] gets original order
        """
        if self.start >= self.size:
            return None, None, None
        src = self.srcdata[self.start:self.start+self.batch]
        ## update `start` for next batch
        self.start += self.batch
        return pad_arrays_keep_invp(src)
