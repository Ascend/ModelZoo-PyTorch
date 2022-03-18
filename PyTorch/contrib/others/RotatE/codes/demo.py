# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import argparse
import torch
import os
from model import KGEModel
import model
from torch.utils.data import DataLoader
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
import torch.nn.functional as F
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Knowledge Graph Embedding Models Demo',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--data_path', type=str, default="data/FB15k-237")
    return parser.parse_args(args)
def train_step(model, optimizer, train_iterator, device):
    '''
    A single train step. Apply back-propation and return the loss
    '''
    # print(args)
    model.train()
    optimizer.zero_grad()
    positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

    if device == 'npu':
        positive_sample = positive_sample.npu()
        negative_sample = negative_sample.npu()
        subsampling_weight = subsampling_weight.npu()

    negative_score = model((positive_sample, negative_sample), mode=mode)

        # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
    negative_score = (F.softmax(negative_score * 1.0, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

    positive_score = model(positive_sample)

    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)


    positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
    negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

    loss = (positive_sample_loss + negative_sample_loss) / 2
    regularization_log = {}
    loss.backward()
    optimizer.step()
    log = {
        **regularization_log,
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item(),
    }

    return log

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
def test(args):
    loc = 'npu:0'
    torch.npu.set_device(loc)
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
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    train_dataset_head = TrainDataset(train_triples, nentity, nrelation, 256, 'head-batch')
    train_dataset_tail = TrainDataset(train_triples, nentity, nrelation, 256, 'tail-batch')
    train_dataloader_head = DataLoader(
        train_dataset_head,
        batch_size=1024,
        shuffle=True,
        num_workers=0,
        collate_fn=TrainDataset.collate_fn
    )

    train_dataloader_tail = DataLoader(
        train_dataset_tail,
        batch_size=1024,
        shuffle=True,
        num_workers=0,
        collate_fn=TrainDataset.collate_fn
    )
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    kge_model_cpu = KGEModel(
        model_name='RotatE',
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=1000,
        gamma=9.0,
        double_entity_embedding=True,
        double_relation_embedding=False
    )
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_0'), map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys()):
        new_key_l = key.split('.')
        if new_key_l[0] == 'module':
            new_key = new_key_l[1]
        else:
            new_key = key
        state_dict[new_key] = state_dict.pop(key)
    kge_model_cpu.load_state_dict(state_dict)
    kge_model_npu = kge_model_cpu.npu()
    optimizer_cpu = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model_cpu.parameters()),
        lr=0.00005
    )
    optimizer_npu = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model_npu.parameters()),
        lr=0.00005
    )
    optimizer_cpu.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Initializing Model Finished')
    #log_cpu = train_step(kge_model_cpu, optimizer_cpu, train_iterator, 'cpu')
    log_npu = train_step(kge_model_npu, optimizer_npu, train_iterator, 'npu')
    #print(log_cpu)
    print(log_npu)


if __name__ == "__main__":
    test(parse_args())
