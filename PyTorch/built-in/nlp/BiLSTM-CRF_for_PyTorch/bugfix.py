# coding: UTF-8
# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
from copy import deepcopy
from os.path import join
from codecs import open
import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from evaluating import Metrics
from data import build_map
from models.util import tensorized
from models.util import sort_by_lengths
from models.util import cal_loss
from models.config import TrainingConfig
from models.config import LSTMConfig
from models.bilstm import BiLSTM


if torch.__version__ >= "1.8":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
from apex import amp
import apex
from torch_npu.utils.profiler import Profile

class NoProfiling(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, word2id, tag2id):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.word2id = word2id
        self.tag2id = tag2id

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def _to_tensor(self, datas):
        batch_sents = [_[0] for _ in datas]
        batch_tags = [_[1] for _ in datas]
        tensorized_sents, lengths = tensorized(batch_sents, self.word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, self.tag2id)
        targets = targets.to(self.device)

        return tensorized_sents, targets, lengths

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, batch_size, device, word2id, tag2id):
    iter = DatasetIterater(dataset, batch_size, device, word2id, tag2id)
    return iter


class BilstmModel(object):
    def __init__(self, vocab_size, out_size, crf=True, args=None):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            crf选择是否添加CRF层"""
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型参数
        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size
        self.vocab_size = vocab_size
        self.out_size = out_size

        self.crf = crf
        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        if not crf:
            self.model = BiLSTM(self.vocab_size, self.emb_size,
                                self.hidden_size, self.out_size).to(self.device)
            self.cal_loss_func = cal_loss
        else:
            self.model = BiLSTM_CRF(self.vocab_size, self.emb_size,
                                    self.hidden_size, self.out_size).to(self.device)

        # 加载训练参数：
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        # 初始化优化器
        self.optimizer = apex.optimizers.NpuFusedAdam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

        if args.amp_opt_level != "O0":
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.amp_opt_level,
                                                        loss_scale="64.0", combine_grad=True)
        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                                   broadcast_buffers=False)

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        batch = self.batch_size
        end_time = time.time()
        for e in range(1, self.epoches + 1):
            self.step = 0
            losses = 0.
            profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)), profile_type=os.getenv("PROFILE_TYPE"))
            for ind in range(0, len(word_lists), batch):
                if self.args.iteration_num != -1 and self.args.iteration_num < (self.step + 1):
                    break
                batch_sents = word_lists[ind:ind + batch]
                batch_tags = tag_lists[ind:ind + batch]
                profiler.start()
                losses += self.train_step(batch_sents,
                                        batch_tags, word2id, tag2id)
                profiler.end()
                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // batch + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f} step_time:{:.6f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step,
                        time.time() - end_time
                    ))
                    losses = 0.
                end_time = time.time()
            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(
                dev_word_lists, dev_tag_lists, word2id, tag2id)
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_ddp(self, word_lists, tag_lists,
                  dev_word_lists, dev_tag_lists,
                  word2id, tag2id):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        train_iter = build_iterator(list(zip(word_lists, tag_lists)), self.batch_size, self.device, word2id, tag2id)
        dev_iter = build_iterator(list(zip(dev_word_lists, dev_tag_lists)), self.batch_size, self.device, word2id,
                                  tag2id)

        batch = self.batch_size
        end_time = time.time()
        profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)), profile_type=os.getenv("PROFILE_TYPE"))
        for e in range(1, self.epoches + 1):
            self.step = 0
            losses = 0.
            for ind, (tensorized_sents, targets, lengths) in enumerate(train_iter):
                self.model.train()
                self.step += 1
                # forward
                profiler.start()
                scores = self.model(tensorized_sents, lengths)

                # 计算损失 更新参数
                self.optimizer.zero_grad()
                loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                profiler.end()
                losses += loss.item()

                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // batch + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f} step_time:{:.6f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step,
                        time.time() - end_time
                    ))
                    losses = 0.
                end_time = time.time()

            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate_ddp(dev_iter, tag2id)
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # 准备数据
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)
        # forward
        scores = self.model(tensorized_sents, lengths)

        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind + self.batch_size]
                batch_tags = dev_tag_lists[ind:ind + self.batch_size]
                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("Saving Model...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def validate_ddp(self, dev_iter, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind, (tensorized_sents, targets, lengths) in enumerate(dev_iter):
                val_step += 1
                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()

            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("Saving Model...")
                self.best_model = self.model
                self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        self.best_model.eval()
        if hasattr(self.best_model, 'module'):
            self.best_model = self.best_model.module
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_p=False, args=None):
    torch.npu.set_device(args.local_rank)
    if args.distributed:
        rank = args.local_rank
        world_size = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        seed = args.seed + dist.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True


    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BilstmModel(vocab_size, out_size, crf=crf, args=args)
    if args.distributed:
        bilstm_model.train_ddp(train_word_lists, train_tag_lists,
                               dev_word_lists, dev_tag_lists, word2id, tag2id)
    else:
        bilstm_model.train(train_word_lists, train_tag_lists,
                           dev_word_lists, dev_tag_lists, word2id, tag2id)

    saving_path = './BiLstm/ckpts/'
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    model_name = "bilstm_crf" if crf else "bilstm"
    torch.save(bilstm_model.model.state_dict(), saving_path + model_name + ".pt")
    if crf:
        print("Totaltime,{}S.".format(int(time.time() - start)))
    else:
        print("Totaltime,{}S.".format(int(time.time() - start)))
    print("Evaluating {}...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_p)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


from models.bilstm import BiLSTM


def forward(self, sents_tensor, lengths):
    emb = self.embedding(sents_tensor)  # [B, L, emb_size]
    rnn_out, _ = self.bilstm(emb)
    scores = self.lin(rnn_out)  # [B, L, out_size]

    return scores


BiLSTM.forward = forward


def build_corpus(split, make_vocab=True, data_dir="./BiLstm/ResumeNER"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists
