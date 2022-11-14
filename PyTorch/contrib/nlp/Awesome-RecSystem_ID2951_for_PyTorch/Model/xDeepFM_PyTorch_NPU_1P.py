# BSD 3-Clause License
#
# Copyright (c) 2022 Huawei Technologies Co., Ltd
# All rights reserved.
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

import re
import os
import math
import torch
import torch.npu
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
import time
import argparse
import torch.npu

from sklearn.metrics import roc_auc_score
from apex import amp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from common.base import Trainer
from common.averageMeter import AverageMeter
from apex.optimizers import NpuFusedAdam


#增加cache
option = {}
option["ACL_OP_COMPILER_CACHE_MODE"] = "enable"
option["ACL_OP_COMPILER_CACHE_DIR"] = "./my_kernel_meta"
print("option:",option)
torch.npu.set_option(option)


EPOCHS = 5
AID_DATA_DIR = '../data/Criteo/forXDeepFM/'
CALCULATE_DEVICE = "npu:5"
DEVICE = torch.device(CALCULATE_DEVICE)
dump_path = '../output/dump'
torch.npu.set_device(CALCULATE_DEVICE)


"""*******************  参数设置  ***************************"""
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=2048, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_backend', default='hccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-bm', '--benchmark', default=0, type=int,
                    metavar='N', help='set benchmark status (default: 1,run benchmark)')
parser.add_argument('--device', default='npu', type=str,
                    help='npu or gpu')
parser.add_argument('--device_num', default=-1, type=int,
                    help='device_num')
parser.add_argument('--device_list', default='0', type=str,
                    help='device id list')
parser.add_argument('--warm_up_epochs', default=0, type=int,
                    help='warm up')
parser.add_argument('--prof', dest='use_prof', action='store_true',
                    help='use prof or not')
parser.add_argument('--distributed', action='store_true',
                    help='use DDP or not')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--npu_device', default='0', type=str,
                    help='specifies the id of the NPU to use')
parser.add_argument('--loss_scale', default=128, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt_level', default='O1', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes')
parser.add_argument('--local_rank', default=0, type=int,
                    help='local rank of the node')
args = parser.parse_args()

npus_per_node = 1
args.batch_size = int(args.batch_size / npus_per_node)


"""
PyTorch implementation of Deep & Cross Network[1]

Reference:
[1] xDeepFM: Combining Explicit and Implicit Feature Interactionsfor Recommender Systems,
    Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie,and Guangzhong Sun
    https://arxiv.org/pdf/1803.05170.pdf
[2] TensorFlow implementation of xDeepFM
    https://github.com/Leavingseason/xDeepFM
[3] PaddlePaddle implemantation of xDeepFM
    https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/xdeepfm
[4] PyTorch implementation of xDeepFM
    https://github.com/qian135/ctr_model_zoo/blob/master/xdeepfm.py
"""


class XDeepFMLayer(nn.Module):
    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, cin_layer_sizes, split_half=True,
                 reg_l1=0.01, reg_l2=1e-5, embedding_size=10):
        super().__init__()  # Python2 下使用 super(XDeepFMLayer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat
        self.num_field = num_field
        self.cin_layer_sizes = cin_layer_sizes
        self.deep_layer_sizes = deep_layer_sizes
        self.embedding_size = embedding_size    # denoted by M
        self.dropout_deep = dropout_deep
        self.split_half = split_half

        self.input_dim = num_field * embedding_size

        # init feature embedding
        feat_embedding = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(feat_embedding.weight)
        self.feat_embedding = feat_embedding

        # Compress Interaction Network (CIN) Part
        cin_layer_dims = [self.num_field] + cin_layer_sizes

        prev_dim, fc_input_dim = self.num_field, 0
        self.conv1ds = nn.ModuleList()
        for k in range(1, len(cin_layer_dims)):
            conv1d = nn.Conv1d(cin_layer_dims[0] * prev_dim, cin_layer_dims[k], 1)
            nn.init.xavier_uniform_(conv1d.weight)
            self.conv1ds.append(conv1d)
            if self.split_half and k != len(self.cin_layer_sizes):
                prev_dim = cin_layer_dims[k] // 2
            else:
                prev_dim = cin_layer_dims[k]
            fc_input_dim += prev_dim

        # Deep Neural Network Part
        all_dims = [self.input_dim] + deep_layer_sizes
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(all_dims[i], all_dims[i + 1]))
            setattr(self, 'batchNorm_' + str(i + 1), nn.BatchNorm1d(all_dims[i + 1]))
            setattr(self, 'dropout_' + str(i + 1), nn.Dropout(dropout_deep[i + 1]))

        # Linear Part
        self.linear = nn.Linear(self.input_dim, 1)

        # output Part
        self.output_layer = nn.Linear(1 + fc_input_dim + deep_layer_sizes[-1], 1)

    def forward(self, feat_index, feat_value, use_dropout=True):
        # get feat embedding
        fea_embedding = self.feat_embedding(feat_index)    # None * F * K
        x0 = fea_embedding

        # Linear Part
        linear_part = self.linear(fea_embedding.reshape(-1, self.input_dim))

        # CIN Part
        x_list = [x0]
        res = []
        for k in range(1, len(self.cin_layer_sizes) + 1):
            # Batch * H_K * D, Batch * M * D -->  Batch * H_k * M * D
            z_k = torch.einsum('bhd,bmd->bhmd', x_list[-1], x_list[0])
            z_k = z_k.reshape(x0.shape[0], x_list[-1].shape[1] * x0.shape[1], x0.shape[2])
            x_k = self.conv1ds[k - 1](z_k)
            x_k = torch.relu(x_k)

            if self.split_half and k != len(self.cin_layer_sizes):
                # x, h = torch.split(x, x.shape[1] // 2, dim=1)
                next_hidden, hi = torch.split(x_k, x_k.shape[1] // 2, 1)
            else:
                next_hidden, hi = x_k, x_k

            x_list.append(next_hidden)
            res.append(hi)

        res = torch.cat(res, dim=1)
        res = torch.sum(res, dim=2)

        # Deep NN Part
        y_deep = fea_embedding.reshape(-1, self.num_field * self.embedding_size)  # None * (F * K)
        if use_dropout:
            y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(1, len(self.deep_layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            if use_dropout:
                y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        # Output Part
        concat_input = torch.cat((linear_part, res, y_deep), dim=1)
        output = self.output_layer(concat_input)
        return output


""" ************************************************************************************ """
"""                                   训练和测试xDeepFM模型                                 """
""" ************************************************************************************ """
def train_xDeepFM_model_demo(device):
    """
    训练DeepFM的方式
    :return:
    """
    train_filelist = ["%s%s" % (AID_DATA_DIR + 'train_data/', x) for x in os.listdir(AID_DATA_DIR + 'train_data/')]
    test_filelist = ["%s%s" % (AID_DATA_DIR + 'test_data/', x) for x in os.listdir(AID_DATA_DIR + 'test_data/')]
    train_file_id = [int(re.sub('[\D]', '', x)) for x in train_filelist]
    train_filelist = [train_filelist[idx] for idx in np.argsort(train_file_id)]

    test_file_id = [int(re.sub('[\D]', '', x)) for x in test_filelist]
    test_filelist = [test_filelist[idx] for idx in np.argsort(test_file_id)]

    featIndex = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_4.pkl2', 'rb'))
    feat_cnt = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_cnt_4.pkl2', 'rb'))

    # 下面的num_feat的长度还需要考虑缺失值的处理而多了一个维度
    xdeepfm = XDeepFMLayer(num_feat=len(featIndex) + 1, num_field=39, dropout_deep=[0, 0, 0, 0, 0],
                            deep_layer_sizes=[400, 400, 400, 400], cin_layer_sizes=[100, 100, 50],
                            embedding_size=16).to(device)
    xdeepfm = xdeepfm.to(device)  # ---------新增
    print("Start Training DeepFM Model!")

    # 定义损失函数还有优化器
    optimizer = NpuFusedAdam(xdeepfm.parameters())

    xdeepfm, optimizer = amp.initialize(xdeepfm, optimizer,
                                        opt_level=args.opt_level,
                                        loss_scale=args.loss_scale,
                                        combine_grad=True)

    # 计数train和test的数据量
    train_item_count = get_in_filelist_item_num(train_filelist)
    test_item_count = get_in_filelist_item_num(test_filelist)

    # 由于数据量过大, 如果使用pytorch的DataSet来自定义数据的话, 会耗时很久, 因此, 这里使用其它方式
    for epoch in range(1, args.epochs + 1):
        train(xdeepfm, train_filelist, train_item_count, featIndex, feat_cnt, device, optimizer, epoch)
        test(xdeepfm, test_filelist, test_item_count, featIndex, feat_cnt, device)


def get_in_filelist_item_num(filelist):
    count = 0
    for fname in filelist:
        with open(fname.strip(), 'r') as fin:
            for _ in fin:
                count += 1
    return count


def test(model, test_filelist, test_item_count, featIndex, feat_cnt, device):
    trainer = Trainer()
    fname_idx = 0
    pred_y, true_y = [], []
    features_idxs, features_values, labels = None, None, None
    test_loss = 0
    with torch.no_grad():
        # 不断地取出数据进行计算
        pre_file_data_count = 0  # 记录在前面已经访问的文件中的数据的数量
        for batch_idx in range(math.ceil(test_item_count / args.batch_size)):
            # 取出当前Batch所在的数据的下标
            st_idx, ed_idx = batch_idx * args.batch_size, (batch_idx + 1) * args.batch_size
            ed_idx = min(ed_idx, test_item_count - 1)

            if features_idxs is None:
                features_idxs, features_values, labels = get_idx_value_label(
                    test_filelist[fname_idx], featIndex, feat_cnt, shuffle=False)

            # 得到在现有文件中的所对应的起始位置及终止位置
            st_idx -= pre_file_data_count
            ed_idx -= pre_file_data_count

            # 如果数据越过当前文件所对应的范围时, 则再读取下一个文件
            if ed_idx <= len(features_idxs):
                batch_fea_idxs = features_idxs[st_idx:ed_idx, :]
                batch_fea_values = features_values[st_idx:ed_idx, :]
                batch_labels = labels[st_idx:ed_idx, :]
            else:
                pre_file_data_count += len(features_idxs)

                # 得到在这个文件内的数据
                batch_fea_idxs_part1 = features_idxs[st_idx::, :]
                batch_fea_values_part1 = features_values[st_idx::, :]
                batch_labels_part1 = labels[st_idx::, :]

                # 得到在下一个文件内的数据
                fname_idx += 1
                ed_idx -= len(features_idxs)
                features_idxs, features_values, labels = get_idx_value_label(
                    test_filelist[fname_idx], featIndex, feat_cnt, shuffle=False)
                batch_fea_idxs_part2 = features_idxs[0:ed_idx, :]
                batch_fea_values_part2 = features_values[0:ed_idx, :]
                batch_labels_part2 = labels[0:ed_idx, :]

                # 将两部分数据进行合并(正常情况下, 数据最多只会在两个文件中)
                batch_fea_idxs = np.vstack((batch_fea_idxs_part1, batch_fea_idxs_part2))
                batch_fea_values = np.vstack((batch_fea_values_part1, batch_fea_values_part2))
                batch_labels = np.vstack((batch_labels_part1, batch_labels_part2))

            # 进行格式转换
            batch_fea_values = torch.from_numpy(batch_fea_values)
            batch_labels = torch.from_numpy(batch_labels)

            idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
            idx = idx.to(device)
            value = batch_fea_values.to(device, dtype=torch.float32)
            target = batch_labels.to(device, dtype=torch.float32)
            output = model(idx, value)

            test_loss += F.binary_cross_entropy_with_logits(output, target).to(device)  # ----------修改

            pred_y.extend(list(output.cpu().numpy()))
            true_y.extend(list(target.cpu().numpy()))

        AUC = 'Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y))
        print(AUC)
        trainer.logger.info(AUC)
        test_loss /= math.ceil(test_item_count / args.batch_size)
        Average_loss = 'Test set: Average loss: {:.5f}'.format(test_loss)
        print(Average_loss)
        trainer.logger.info(Average_loss)


def train(model, train_filelist, train_item_count, featIndex, feat_cnt, device, optimizer, epoch, use_reg_l2=True):
    trainer = Trainer()
    fname_idx = 0
    features_idxs, features_values, labels = None, None, None

    # 依顺序来遍历访问
    pre_file_data_count = 0  # 记录在前面已经访问的文件中的数据的数量
    flag = 0
    for batch_idx in range(math.ceil(train_item_count / args.batch_size)):
        start_time2 = time.time()
        # 计算开始时间
        if batch_idx > (math.ceil(train_item_count / args.batch_size)) * 0.1 and flag == 0:
            start_time = time.time()
            flag = 1
        # 得到当前Batch所要取的数据的起始及终止下标
        st_idx, ed_idx = batch_idx * args.batch_size, (batch_idx + 1) * args.batch_size
        ed_idx = min(ed_idx, train_item_count - 1)

        if features_idxs is None:
            features_idxs, features_values, labels = get_idx_value_label(train_filelist[fname_idx], featIndex, feat_cnt)

        # 得到在现有文件中的所对应的起始位置及终止位置
        st_idx -= pre_file_data_count
        ed_idx -= pre_file_data_count

        # 如果数据越过当前文件所对应的范围时, 则再读取下一个文件
        if ed_idx < len(features_idxs):
            batch_fea_idxs = features_idxs[st_idx:ed_idx, :]
            batch_fea_values = features_values[st_idx:ed_idx, :]
            batch_labels = labels[st_idx:ed_idx, :]
        else:
            pre_file_data_count += len(features_idxs)

            # 得到在这个文件内的数据
            batch_fea_idxs_part1 = features_idxs[st_idx::, :]
            batch_fea_values_part1 = features_values[st_idx::, :]
            batch_labels_part1 = labels[st_idx::, :]

            # 得到在下一个文件内的数据
            fname_idx += 1
            ed_idx -= len(features_idxs)
            features_idxs, features_values, labels = get_idx_value_label(train_filelist[fname_idx], featIndex, feat_cnt)
            batch_fea_idxs_part2 = features_idxs[0:ed_idx, :]
            batch_fea_values_part2 = features_values[0:ed_idx, :]
            batch_labels_part2 = labels[0:ed_idx, :]

            # 将两部分数据进行合并(正常情况下, 数据最多只会在两个文件中)
            batch_fea_idxs = np.vstack((batch_fea_idxs_part1, batch_fea_idxs_part2))
            batch_fea_values = np.vstack((batch_fea_values_part1, batch_fea_values_part2))
            batch_labels = np.vstack((batch_labels_part1, batch_labels_part2))

        # 进行格式转换
        batch_fea_values = torch.from_numpy(batch_fea_values)
        batch_labels = torch.from_numpy(batch_labels)

        idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
        idx = idx.to(device)
        value = batch_fea_values.to(device, dtype=torch.float32)
        target = batch_labels.to(device, dtype=torch.float32)

        start_time3 = time.time()
        optimizer.zero_grad()
        output = model(idx, value)
        loss = F.binary_cross_entropy_with_logits(output, target).to(device)

        if use_reg_l2:
            regularization_loss = 0
            for param in model.parameters():
                # regularization_loss += model.reg_l1 * torch.sum(torch.abs(param))
                regularization_loss += model.reg_l2 * torch.sum(torch.pow(param, 2))
            loss += regularization_loss

        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()
        print("NPU exectime: ", start_time3-start_time2, time.time() - start_time2)
        if batch_idx % 1000 == 0:
            screen = 'Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(idx), train_item_count,
                100. * batch_idx / math.ceil(int(train_item_count / args.batch_size)), loss.item())
            print(screen)
            trainer.logger.info(screen)
    epoch_time = time.time() - start_time
    fps = 'FPS: {:.4f}'.format(args.batch_size / epoch_time)
    trainer.logger.info(fps)


def get_idx_value_label(fname, featIndex, feat_cnt, shuffle=True):
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    def _process_line(line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []

        for idx in continuous_range_:
            key = 'I' + str(idx)
            val = features[idx]

            if val == '':
                feat = str(key) + '#' + 'absence'
            else:
                val = int(float(val))
                if val > 2:
                    val = int(math.log(float(val)) ** 2)
                else:
                    val = 'SP' + str(val)
                feat = str(key) + '#' + str(val)

            feat_idx.append(featIndex[feat])
            feat_value.append(1)

        for idx in categorical_range_:
            key = 'C' + str(idx - 13)
            val = features[idx]

            if val == '':
                feat = str(key) + '#' + 'absence'
            else:
                feat = str(key) + '#' + str(val)
            if feat_cnt[feat] > 4:
                feat = feat
            else:
                feat = str(key) + '#' + str(feat_cnt[feat])

            feat_idx.append(featIndex[feat])
            feat_value.append(1)
        return feat_idx, feat_value, [int(features[0])]

    features_idxs, features_values, labels = [], [], []
    with open(fname.strip(), 'r') as fin:
        for line in fin:
            feat_idx, feat_value, label = _process_line(line)
            features_idxs.append(feat_idx)
            features_values.append(feat_value)
            labels.append(label)

    features_idxs = np.array(features_idxs)
    features_values = np.array(features_values)
    labels = np.array(labels).astype(np.int32)

    # 进行shuffle
    if shuffle:
        idx_list = np.arange(len(features_idxs))
        np.random.shuffle(idx_list)

        features_idxs = features_idxs[idx_list, :]
        features_values = features_values[idx_list, :]
        labels = labels[idx_list, :]
    return features_idxs, features_values, labels


if __name__ == '__main__':
    torch.npu.set_device(CALCULATE_DEVICE)
    train_xDeepFM_model_demo(DEVICE)
