# coding: UTF-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
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
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import torch.distributed as dist
import os
from apex import amp
import apex

if torch.__version__ >= '1.8':
    import torch_npu
    
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# DDP argument.
parser.add_argument('--dist_backend', default='hccl', type=str, help='hccl for npu, must!')
parser.add_argument('--addr', default='127.0.0.1', type=str, help='hccl ip address')
parser.add_argument('--Port', default='888888', type=str, help='hccl ip Port')
parser.add_argument('--world_size', default=1, type=int, help='ddp world size')
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--num_epochs', default=20, type=int, help='number of train epoch')
parser.add_argument('--distributed', action="store_true", help='distributed')
parser.add_argument('--data_path', default='THUCNews', type=str, help='data path')
args = parser.parse_args()


def main():
    dataset = args.data_path  # 数据集
    print("args.world_size = ", args.world_size)
    os.environ["MASTER_ADDR"] = args.addr
    os.environ["MASTER_PORT"] = args.Port
    print("args.addr = ", args.addr)
    print("args.Port = ", args.Port)
    args.rank = 0

    if args.distributed:
        args.device = 'npu:%d' % args.local_rank
        torch.npu.set_device(args.device)
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.local_rank)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = torch.distributed.get_rank()

    else:
        args.device = f'npu:{args.local_rank}'
        torch.npu.set_device(args.device)

    print('local_rank {}'.format(args.local_rank))

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    config.device = args.device
    config.world_size = args.world_size
    config.local_rank = args.local_rank
    config.batch_size = config.batch_size * config.world_size
    config.distributed = args.distributed
    config.num_epochs = args.num_epochs

    np.random.seed(666)
    torch.manual_seed(666)
    torch.npu.manual_seed_all(666)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)

    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=config.learning_rate)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale="dynamic", combine_grad=True,master_weights=True)
 

    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)  
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter, optimizer)

if __name__ == '__main__':
    main()
