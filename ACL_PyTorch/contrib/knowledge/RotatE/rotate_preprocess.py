# Copyright 2022 Huawei Technologies Co., Ltd
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


from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import sys
sys.path.append(r'KnowledgeGraphEmbedding/codes/')
from run import read_triple
import dataloader


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Data preprocessing for Knowledge Graph Embedding Models',
        usage='RotatE_preprocess.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_path', type=str,
                        default='./KnowledgeGraphEmbedding/data/FB15k-237')
    parser.add_argument('--test_batch_size', default=6,
                        type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--output_path', default='bin', type=str)
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
    parser.add_argument('--nentity', type=int, default=0,
                        help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0,
                        help='DO NOT MANUALLY SET')
    arg = parser.parse_args(args)
    arg.output_head_post = os.path.join(arg.output_path, arg.output_head_post)
    arg.output_tail_post = os.path.join(arg.output_path, arg.output_tail_post)
    arg.output_head_pos = os.path.join(arg.output_path, arg.output_head_pos)
    arg.output_head_neg = os.path.join(arg.output_path, arg.output_head_neg)
    arg.output_head_mode = os.path.join(arg.output_path, arg.output_head_mode)
    arg.output_head_pp = os.path.join(arg.output_path, arg.output_head_pp)
    arg.output_head_np = os.path.join(arg.output_path, arg.output_head_np)
    arg.output_tail_pos = os.path.join(arg.output_path, arg.output_tail_pos)
    arg.output_tail_neg = os.path.join(arg.output_path, arg.output_tail_neg)
    arg.output_tail_mode = os.path.join(arg.output_path, arg.output_tail_mode)
    arg.output_tail_pp = os.path.join(arg.output_path, arg.output_tail_pp)
    arg.output_tail_np = os.path.join(arg.output_path, arg.output_tail_np)
    return arg


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

    args.nentity = len(entity2id)
    args.nrelation = len(relation2id)

    train_triples = read_triple(os.path.join(
        args.data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(
        args.data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(
        args.data_path, 'test.txt'), entity2id, relation2id)

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

    for dirs in [args.output_head_pos, args.output_head_neg, args.output_head_mode, args.output_head_pp,
                 args.output_tail_pos, args.output_tail_neg, args.output_tail_mode, args.output_tail_pp]:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    for index, (positive_sample, negative_sample, filter_bias, mode) in enumerate(tqdm(test_dataloader_head, desc="Preprocessing head data...")):
        filename = f'bin{args.test_batch_size * index}-{args.test_batch_size * (index + 1) - 1}'

        save_path_pos = os.path.join(args.output_head_pos, f'{filename}.bin')
        save_path_pos_txt = os.path.join(args.output_head_pp, f'{filename}.txt')
        positive_sample.long().numpy().tofile(save_path_pos)
        np.savetxt(save_path_pos_txt, positive_sample.long().numpy())

        save_path_neg = os.path.join(args.output_head_neg, f'{filename}.bin')
        negative_sample.int().numpy().tofile(save_path_neg)

        save_post_dir = str(args.output_head_post)
        if not os.path.exists(save_post_dir):
            os.makedirs(save_post_dir)
        save_path_post = os.path.join(save_post_dir, f'{filename}.txt')
        np.savetxt(save_path_post, filter_bias.numpy())

    for index, (positive_sample, negative_sample, filter_bias, mode) in enumerate(tqdm(test_dataloader_tail, desc="Preprocessing tail data...")):
        filename = f'bin{args.test_batch_size * index}-{args.test_batch_size * (index + 1) - 1}'

        save_path_pos = os.path.join(args.output_tail_pos, f'{filename}.bin')
        save_path_pos_txt = os.path.join(args.output_tail_pp, f'{filename}.txt')
        positive_sample.long().numpy().tofile(save_path_pos)
        np.savetxt(save_path_pos_txt, positive_sample.long().numpy())

        save_path_neg = os.path.join(args.output_tail_neg, f'{filename}.bin')
        negative_sample.int().numpy().tofile(save_path_neg)

        save_post_dir = str(args.output_tail_post)
        if not os.path.exists(save_post_dir):
            os.makedirs(save_post_dir)
        save_path_post = os.path.join(save_post_dir, f'{filename}.txt')
        np.savetxt(save_path_post, filter_bias.numpy())


if __name__ == '__main__':
    main(parse_args())
