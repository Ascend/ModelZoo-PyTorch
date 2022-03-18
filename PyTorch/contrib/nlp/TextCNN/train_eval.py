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
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from utils import get_time_dif
from apex import amp

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, model, train_iter, dev_iter, test_iter, optimizer):

    start_time = time.time()
    batch_time = AverageMeter()
    model.train()

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False


    end = time.time()
    for epoch in range(config.num_epochs):
        if config.local_rank == 0:
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        scheduler.step()
        for i, (trains, labels) in enumerate(train_iter):
            # if i == 10:
                # with torch.autograd.profiler.profile(record_shapes=True,use_npu=True) as prof:
                    # outputs = model(trains)
                    # model.zero_grad()
                    # loss = F.cross_entropy(outputs, labels)
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                        # scaled_loss.backward()
                    # optimizer.step()
                # prof.export_chrome_trace("textCNN1208_1P.prof")
                # with torch.npu.profile("cann_profiling"):
                    # outputs = model(trains)
                    # model.zero_grad()
                    # loss = F.cross_entropy(outputs, labels)
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                        # scaled_loss.backward()
                    # optimizer.step()
                # import sys
                # sys.exit()


            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if i == 2:
                batch_time.reset()
            batch_time.update(time.time() - end)
            fps_tmp = config.batch_size / batch_time.val
            fps_avg = config.batch_size / batch_time.avg
            if total_batch % 1 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    if config.local_rank == 0:
                        torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                if config.local_rank == 0:
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}, fps_tmp[fps_avg]: {6:>4.3}[{7:>4.3}] {8}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, fps_tmp, fps_avg, improve))
                model.train()
            total_batch += 1
            end = time.time()

    if config.local_rank == 0:
        test(config, model, test_iter, config.local_rank)


def test(config, model, test_iter, local_rank):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    if local_rank == 0:
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
