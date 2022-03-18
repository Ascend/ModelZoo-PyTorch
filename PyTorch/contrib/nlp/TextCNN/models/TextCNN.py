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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('npu' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


# class Model(nn.Module):
    # def __init__(self, config):
        # super(Model, self).__init__()
        # if config.embedding_pretrained is not None:
            # self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=True)
        # else:
            # self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # self.convs00 = nn.Conv2d(1, 256, (2, 30), (1, 30))
        # self.convs01 = nn.Conv2d(1, 256, (3, 30), (1, 30))
        # self.convs02 = nn.Conv2d(1, 256, (4, 30), (1, 30))

        # self.convs10 = nn.Conv2d(256, 256, (1, 10)) 
        # self.convs11 = nn.Conv2d(256, 256, (1, 10)) 
        # self.convs12 = nn.Conv2d(256, 256, (1, 10))

        # self.pools0 = nn.MaxPool2d((31, 1))
        # self.pools1 = nn.MaxPool2d((30, 1))
        # self.pools2 = nn.MaxPool2d((29, 1))
        
        # self.flatten0 = nn.Flatten()
        # self.flatten1 = nn.Flatten()
        # self.flatten2 = nn.Flatten()

        # self.dropout = nn.Dropout(config.dropout)
        # self.fc = nn.Linear(768, config.num_classes)

    # def conv_and_pool(self, x, conv0, conv1, pool, flatten):
        # x = flatten(pool(F.relu(conv1(conv0(x)))))
        # return x

    # def forward(self, x):
        # out = self.embedding(x)
        # out = out.unsqueeze(1)
        # out = torch.cat([
            # self.conv_and_pool(out, self.convs00, self.convs10, self.pools0, self.flatten0),
            # self.conv_and_pool(out, self.convs01, self.convs11, self.pools1, self.flatten1),
            # self.conv_and_pool(out, self.convs02, self.convs12, self.pools2, self.flatten2),
        # ], 1)
        # out = self.dropout(out)
        # out = self.fc(out)
        # return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs0 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed // 10), (1, config.embed // 10)) for k in config.filter_sizes]
        )
        self.convs1 = nn.ModuleList([nn.Conv2d(config.num_filters, config.num_filters, (1, 10))] * 3)
        self.pools = nn.ModuleList([nn.MaxPool1d(config.pad_size - k + 1) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv0, conv1, pool):
        # x = F.relu(conv1(conv0(x))).squeeze(3)
        # x = pool(x).squeeze(2)
        # return x
        x0 = conv0(x)
        x1 = conv1(x0)
        x_relu = F.relu(x1)
        x_sq3 = x_relu.squeeze(3)
        x_pool = pool(x_sq3)
        x_sq2 = x_pool.squeeze(2)
        return x_sq2


    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv0, conv1, pool) for conv0, conv1, pool in zip(self.convs0, self.convs1, self.pools)], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
