# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
sys.path.append(r'KnowledgeGraphEmbedding/codes/')
import logging
import os
import io
import pdb
import torch
import numpy as np

import time
from torch.utils.data import DataLoader
import dataloader

nowTime = time.strftime('%Y%m%d', time.localtime(time.time()))

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_path', type=str, default='./KnowledgeGraphEmbedding/data/FB15k-237')
    parser.add_argument('--test_batch_size', default=6, type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--output_path', default='bin/', type=str)
    parser.add_argument('--output_head_post', default='head/post', type=str)
    parser.add_argument('--output_tail_post', default='tail/post', type=str)
    parser.add_argument('--output_head_pos', default='head/pos', type=str)
    parser.add_argument('--output_head_neg', default='head/neg', type=str)
    parser.add_argument('--output_head_mode', default='head/mode', type=str)
    parser.add_argument('--output_head_pp', default='head/possamp', type=str)
    parser.add_argument('--output_head_np', default='head/negsamp', type=str)
    parser.add_argument('--output_tail_pos', default='tail/pos', type=str)
    parser.add_argument('--output_tail_neg', default='tail/neg', type=str)
    parser.add_argument('--output_tail_mode', default='tail/mode', type=str)
    parser.add_argument('--output_tail_pp', default='tail/possamp', type=str)
    parser.add_argument('--output_tail_np', default='tail/negsamp', type=str)
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    arg =  parser.parse_args(args)
    arg.output_head_post = arg.output_path + arg.output_head_post
    arg.output_tail_post = arg.output_path + arg.output_tail_post
    arg.output_head_pos = arg.output_path + arg.output_head_pos
    arg.output_head_neg = arg.output_path + arg.output_head_neg
    arg.output_head_mode = arg.output_path + arg.output_head_mode
    arg.output_head_pp = arg.output_path + arg.output_head_pp
    arg.output_head_np = arg.output_path + arg.output_head_np
    arg.output_tail_pos = arg.output_path + arg.output_tail_pos
    arg.output_tail_neg = arg.output_path + arg.output_tail_neg
    arg.output_tail_mode = arg.output_path + arg.output_tail_mode
    arg.output_tail_pp = arg.output_path + arg.output_tail_pp
    arg.output_tail_np = arg.output_path + arg.output_tail_np
    return arg

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def to_numpy32(tensor):
    return tensor.detach().cpu().numpy().astype(np.int32) if tensor.requires_grad else tensor.cpu().numpy().astype(np.int32)

def to_numpy64(tensor):
    return tensor.detach().cpu().numpy().astype(np.int64) if tensor.requires_grad else tensor.cpu().numpy().astype(np.int64)

def main(args):

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)


    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation


    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    test_dataloader_head = DataLoader(
        dataloader.TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'head-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=dataloader.TestDataset.collate_fn
    )

    test_dataloader_tail = DataLoader(
        dataloader.TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'tail-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=dataloader.TestDataset.collate_fn
    )

    test_dataset_list = [test_dataloader_head, test_dataloader_tail]
    # test_dataset_list = [test_dataloader_tail]
    for test_dataset in test_dataset_list:
        for index, value in enumerate(test_dataset):
            if(value[0].shape[0] == args.test_batch_size):
                batch_pos = value[0]
                batch_pos = to_numpy64(batch_pos)

                batch_neg = value[1]
                batch_neg = to_numpy32(batch_neg)
                batch_ite = value[2].numpy()
                batch_mode = value[3]

                print('preprocessing ' + str(index))

                if not os.path.exists(str(args.output_head_pos)):
                    os.makedirs(str(args.output_head_pos))
                if not os.path.exists(str(args.output_head_neg)):
                    os.makedirs(str(args.output_head_neg))
                if not os.path.exists(str(args.output_head_mode)):
                    os.makedirs(str(args.output_head_mode))
                if not os.path.exists(str(args.output_head_pp)):
                    os.makedirs(str(args.output_head_pp))
                if not os.path.exists(str(args.output_tail_pos)):
                    os.makedirs(str(args.output_tail_pos))
                if not os.path.exists(str(args.output_tail_neg)):
                    os.makedirs(str(args.output_tail_neg))
                if not os.path.exists(str(args.output_tail_mode)):
                    os.makedirs(str(args.output_tail_mode))
                if not os.path.exists(str(args.output_tail_pp)):
                    os.makedirs(str(args.output_tail_pp))


                if batch_mode == 'head-batch':
                    save_path_pos = str(args.output_head_pos) + '/bin' + str(int(args.test_batch_size) * index) + '-' + str(
                        int(args.test_batch_size) * (index + 1) - 1) + '.bin'
                    save_path_pos_txt = str(args.output_head_pp) + '/bin' + str(int(args.test_batch_size) * index) + '-' + str(
                        int(args.test_batch_size) * (index + 1) - 1) + '.txt'
                    batch_pos.tofile(str(save_path_pos))
                    np.savetxt(save_path_pos_txt, batch_pos)

                    save_path_neg = str(args.output_head_neg) + '/bin' + str(int(args.test_batch_size) * index) + '-' + str(
                        int(args.test_batch_size) * (index + 1) - 1) + '.bin'
                    batch_neg.tofile(str(save_path_neg))

                    save_post_dir = str(args.output_head_post)
                    if not os.path.exists(save_post_dir):
                        os.makedirs(save_post_dir)
                    save_path_post = save_post_dir + '/bin' + str(int(args.test_batch_size) * index) + '-' + str(
                        int(args.test_batch_size) * (index + 1) - 1) + '.txt'
                    np.savetxt(save_path_post, batch_ite)
                    print(index, str(save_path_post), "save done!")
                    print("----------------head---next-----------------------------")

                if batch_mode == 'tail-batch':

                    save_path_pos = str(args.output_tail_pos) + '/bin' + str(int(args.test_batch_size) * index) + '-' + str(
                        int(args.test_batch_size) * (index + 1) - 1) + '.bin'
                    save_path_pos_txt = str(args.output_tail_pp) + '/bin' + str(int(args.test_batch_size) * index) + '-' + str(
                        int(args.test_batch_size) * (index + 1) - 1) + '.txt'
                    batch_pos.tofile(str(save_path_pos))
                    np.savetxt(save_path_pos_txt, batch_pos)

                    save_path_neg = str(args.output_tail_neg) + '/bin' + str(int(args.test_batch_size) * index) + '-' + str(
                        int(args.test_batch_size) * (index + 1) - 1) + '.bin'
                    batch_neg.tofile(str(save_path_neg))

                    print(index, str(save_path_neg), "save done!")

                    save_post_dir = str(args.output_tail_post)
                    if not os.path.exists(save_post_dir):
                        os.makedirs(save_post_dir)
                    save_path_post = save_post_dir + '/bin' + str(int(args.test_batch_size) * index) + '-' + str(
                        int(args.test_batch_size) * (index + 1) - 1) + '.txt'
                    np.savetxt(save_path_post, batch_ite)
                    print(index, str(save_path_post), "save done!")
                    print("---------------tail----next-----------------------------")


if __name__ == '__main__':
    main(parse_args())
