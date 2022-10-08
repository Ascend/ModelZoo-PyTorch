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
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""
import os
import gc
import numpy as np
from tqdm import tqdm

import faiss

__all__ = ['faiss_search_approx_knn']

class faiss_index_wrapper():
    def __init__(self,
                 target,
                 nprobe=128,
                 index_factory_str=None,
                 verbose=False,
                 mode='proxy',
                 using_gpu=True):
        self._res_list = []

        num_gpu = faiss.get_num_gpus()
        print('[faiss gpu] #GPU: {}'.format(num_gpu))

        size, dim = target.shape
        assert size > 0, "size: {}".format(size)
        index_factory_str = "IVF{},PQ{}".format(
            min(8192, 16 * round(np.sqrt(size))),
            32) if index_factory_str is None else index_factory_str
        cpu_index = faiss.index_factory(dim, index_factory_str)
        cpu_index.nprobe = nprobe

        if mode == 'proxy':
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False

            index = faiss.IndexProxy()
            for i in range(num_gpu):
                res = faiss.StandardGpuResources()
                self._res_list.append(res)
                sub_index = faiss.index_cpu_to_gpu(
                    res, i, cpu_index, co) if using_gpu else cpu_index
                index.addIndex(sub_index)
        elif mode == 'shard':
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
            co.shard = True
            index = faiss.index_cpu_to_all_gpus(cpu_index,
                                                co,
                                                ngpu=num_gpu)
        else:
            raise KeyError("Unknown index mode")

        index = faiss.IndexIDMap(index)
        index.verbose = verbose

        # get nlist to decide how many samples used for training
        nlist = int(float([
            item for item in index_factory_str.split(",") if 'IVF' in item
        ][0].replace("IVF", "")))

        # training
        if not index.is_trained:
            indexes_sample_for_train = np.random.randint(
                0, size, nlist * 256)
            index.train(target[indexes_sample_for_train])

        # add with ids
        target_ids = np.arange(0, size)
        index.add_with_ids(target, target_ids)
        self.index = index

    def search(self, *args, **kargs):
        return self.index.search(*args, **kargs)

    def __del__(self):
        self.index.reset()
        del self.index
        for res in self._res_list:
            del res


def batch_search(index, query, k, bs, verbose=False):
    n = len(query)
    dists = np.zeros((n, k), dtype=np.float32)
    nbrs = np.zeros((n, k), dtype=np.int64)

    for sid in tqdm(range(0, n, bs),
                    desc="faiss searching...",
                    disable=not verbose):
        eid = min(n, sid + bs)
        dists[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], k)
    return dists, nbrs


def faiss_search_approx_knn(query,
                            target,
                            k,
                            nprobe=128,
                            bs=int(1e6),
                            index_factory_str=None,
                            verbose=False):
    index = faiss_index_wrapper(target,
                                nprobe=nprobe,
                                index_factory_str=index_factory_str,
                                verbose=verbose)
    dists, nbrs = batch_search(index, query, k=k, bs=bs, verbose=verbose)

    del index
    gc.collect()
    return dists, nbrs
