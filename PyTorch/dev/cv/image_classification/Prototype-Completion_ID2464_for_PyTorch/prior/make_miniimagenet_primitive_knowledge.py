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
import argparse
import json
import pickle

from nltk.corpus import wordnet as wn
import torch
import numpy as np

from prior.glove import GloVe
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


def constructedges(s, syns2id):
    edges = []
    for k, vs in s.items():
        for v in vs:
            edges.append((syns2id[k], syns2id[v]))
    return edges

def make_attribute_node(syns, train_nodes, val_nodes, test_nodes):
    syns_paths = []
    syns_len = len(syns)
    for i in range(syns_len):
        if i == 96:
            print('stop')
        paths =  syns[i].hypernym_paths()
        syns_paths.extend(paths)
        print('number {}: {}'.format(i, [path[4].lemma_names for path in paths]))
    for i in range(20):
        try:
            syns_i = [path[i] for path in syns_paths]
            print('number {}: {}'.format(i, len(set(syns_i))))
        except:
            print('number {}: {}'.format(i, 0))

    attrbute = []
    syns_attrbute = {}
    syns_len = len(syns)
    for i in range(syns_len):
        attrbute.append(syns[i])
    for i in range(syns_len):
        syns_attrbute[syns[i]] = []
        have_attri = False
        for paths in syns[i].hypernym_paths():
            for sys in paths:
                parts = sys.part_meronyms()
                # parts = sys.substance_meronyms()
                attrbute.extend(parts)
                syns_attrbute[syns[i]].extend(parts)
                if len(parts) != 0:
                    have_attri = True
        if have_attri:
            print('number {}: {}'.format(i, 'attribute'))
        else:
            print('number {}: {}'.format(i, 'no attribute'))
        syns_attrbute[syns[i]] = list(set(syns_attrbute[syns[i]]))
    attrbute = list(set(attrbute))

    # 鑾峰緱姣忎釜鏁版嵁闆嗕笅鐨勫睘鎬
    train_attribute = []
    for i in range(len(train_nodes)):
        train_attribute.extend(syns_attrbute[train_nodes[i]])
    train_attribute = list(set(train_attribute))

    val_attribute = []
    for i in range(len(val_nodes)):
        val_attribute.extend(syns_attrbute[val_nodes[i]])
    val_attribute = list(set(val_attribute))

    test_attribute = []
    for i in range(len(test_nodes)):
        test_attribute.extend(syns_attrbute[test_nodes[i]])
    test_attribute = list(set(test_attribute))

    attrbute_rm = []
    syns_attrbute_rm = {}
    train_attribute_rm = []
    val_attribute_rm = []
    test_attribute_rm = []

    for attr in train_attribute:
        if attr in val_attribute or attr in test_attribute:
            train_attribute_rm.append(attr)

    for attr in val_attribute:
        if attr in train_attribute:
            val_attribute_rm.append(attr)

    for attr in test_attribute:
        if attr in train_attribute:
            test_attribute_rm.append(attr)

    attrbute_rm = syns + list(set(train_attribute_rm + val_attribute_rm + test_attribute_rm))
    for syn in syns:
        attrs = syns_attrbute[syn]
        syns_attrbute_rm[syn] = [attr for attr in attrs if attr in list(set(train_attribute_rm + val_attribute_rm + test_attribute_rm))] + [syn, ]

    return attrbute_rm, syns_attrbute_rm

if __name__ == '__main__':
    output = '../data/mini_imagenet_part_prior_train.pickle'
    train_class_name_file = '../data/mini_imagenet_catname2label_train.pickle'
    val_class_name_file = '../data/mini_imagenet_catname2label_val.pickle'
    test_class_name_file = '../data/mini_imagenet_catname2label_test.pickle'
    print('making graph ...')

    with open(train_class_name_file, 'rb') as handle:
        catname2label_train = pickle.load(handle)
    wnids_train = catname2label_train.keys()
    with open(val_class_name_file, 'rb') as handle:
        catname2label_val = pickle.load(handle)
    wnids_val = catname2label_val.keys()
    with open(test_class_name_file, 'rb') as handle:
        catname2label_test = pickle.load(handle)
    wnids_test = catname2label_test.keys()

    # all_wnids = list(wnids_train) + list(wnids_val)
    all_wnids = list(wnids_train) + list(wnids_val) + list(wnids_test)
    all_wnids = list(np.unique(all_wnids))

    all_nodes = list(map(getnode, all_wnids))
    train_nodes = list(map(getnode, list(wnids_train)))
    val_nodes = list(map(getnode, list(wnids_val)))
    test_nodes = list(map(getnode, list(wnids_test)))
    # all_set = set(all_nodes)

    attribute_node, node_attribute_dict = make_attribute_node(all_nodes,
                                                              train_nodes,
                                                              val_nodes,
                                                              test_nodes)

    wnids = list(map(getwnid, attribute_node))
    wnids2id = {wnid:i for i, wnid in enumerate(wnids)}
    id2wnids = {v: k for k, v in wnids2id.items()}
    syns2id = {getnode(wnid): i for i, wnid in id2wnids.items()}
    edges = constructedges(node_attribute_dict, syns2id)
    class_attribute_id_dict = {syns2id[k]: [syns2id[v] for v in vs] for k, vs in node_attribute_dict.items()}
    attribute_id_class_dict = {}
    for k, vs in class_attribute_id_dict.items():
        for v in vs:
            if v in attribute_id_class_dict.keys():
                attribute_id_class_dict[v].append(k)
            else:
                attribute_id_class_dict[v] = [k, ]

    print('making glove embedding ...')

    glove = GloVe('/extend/zhangbq/code/datasets/few_shot_data/glove.840B.300d.txt')
    vectors = []
    num = 0
    for wnid in wnids:
        # print(getnode(wnid).lemma_names())
        vectors.append(glove[getnode(wnid).lemma_names()])
        if torch.sum(torch.abs(vectors[-1])) == 0:
            print('wnid: {}锛寋}'.format(wnid, getnode(wnid).lemma_names()))
            num+=1
    print(num)
    vectors = torch.stack(vectors)

    print('dumping ...')

    obj = {}
    obj['all_wnids'] = all_wnids
    obj['wnids_train'] = list(wnids_train)
    obj['wnids_val'] = list(wnids_val)
    obj['wnids_test'] = list(wnids_test)
    obj['wnids'] = wnids
    obj['vectors'] = vectors.tolist()
    obj['wnids2id'] = wnids2id
    obj['id2wnids'] = id2wnids
    obj['class_attribute_id_dict'] = class_attribute_id_dict
    obj['attribute_id_class_dict'] = attribute_id_class_dict
    obj['edges'] = edges
    with open(output, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

