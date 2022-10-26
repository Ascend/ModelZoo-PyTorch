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
import os
import time
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
if torch.__version__ >= "1.8":
    import torch_npu
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms as trans
from tensorboardX import SummaryWriter
import apex
from apex import amp

from data.data_pipe import get_train_loader, get_val_pair
from model import Backbone, Arcface, MobileFaceNet, l2_norm
from verifacation import evaluate
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras

plt.switch_backend('agg')


# nohup时不会延迟打印
def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()

    return new_print


print = flush_print(print)


def prepare_eval_data(data_folder):
    """
    when finetune, use it
    """
    data_label_path = os.path.join(data_folder, 'label.txt')
    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    with open(data_label_path, 'r', encoding='utf-8') as f:
        data_label_list = f.readlines()
    data_label_list = [i.strip().split(' ') for i in data_label_list if len(i) > 4]

    carray, issame = [], []
    for data_label in data_label_list:
        idx, label = data_label
        img_ids = next(os.walk(data_folder / idx))[2]
        print(img_ids)
        if len(img_ids) != 2:
            raise ValueError(f'please check eval dataset, {idx} contains multiple images ')
        for img_id in img_ids:
            img = Image.open(os.path.join(data_folder / idx / img_id))
            img = img.convert('RGB')
            img = transform(img)
            carray.append(img)

        if label == 'True':
            issame.append(True)
        else:
            issame.append(False)

    carray = torch.stack(carray)
    issame = np.array(issame, dtype=np.bool)
    return carray, issame


class face_learner(object):
    def __init__(self, conf, inference=False):
        # 分布式环境下只在主节点下输出日志，单卡则正常输出
        if conf.is_master_node:
            print(conf)

        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        self.model = self.model.to(conf.device)
        # 去除分布式模型权重 带来的影响
        self.model_without_ddp = self.model

        if not inference:
            # 学习率调整节点列表
            self.milestones = conf.milestones
            # 训练集
            self.loader, self.class_num = get_train_loader(conf)

            # 验证数据集
            if conf.is_finetune:
                self.lfw, self.lfw_issame = prepare_eval_data(conf.emore_folder / conf.eval_data_mode)
            else:
                self.lfw, self.lfw_issame = get_val_pair(conf.emore_folder, conf.eval_data_mode)

            self.writer = SummaryWriter(conf.log_path)
            # 开始epoch，用于resume
            self.start_epoch = conf.start_epoch
            # 迭代次数全局变量，用于提前截至，打印loss
            self.step = 1
            # 训练头
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            # 去除分布式训练下 模型权重名称冗余带来的影响
            self.head_without_ddp = self.head
            # 创建loss
            self.loss_func = conf.ce_loss.to(conf.device)
            # 解析模型参数
            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            # 创建optimizer
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            elif conf.use_amp and conf.device_type == 'npu':
                self.optimizer = apex.optimizers.NpuFusedSGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)

            # 分布式环境下只在主节点下输出日志，单卡则正常输出
            if conf.is_master_node:
                print(self.optimizer)
                print('optimizers generated')

            # 打印步长
            self.board_loss_every = min(500, conf.max_iter // 10) if conf.max_iter != -1 else len(self.loader) // 10

            # apex amp 半精度优化
            if conf.use_amp:
                if conf.device_type == 'npu':
                    [self.model, self.head], self.optimizer = amp.initialize([self.model, self.head],
                                                                             self.optimizer,
                                                                             opt_level=conf.opt_level,
                                                                             loss_scale=conf.loss_scale,
                                                                             combine_grad=True)
                else:
                    [self.model, self.head], self.optimizer = amp.initialize([self.model, self.head],
                                                                             self.optimizer,
                                                                             opt_level=conf.opt_level,
                                                                             loss_scale=conf.loss_scale)

            # 分布式训练
            if conf.distributed:
                if conf.use_amp and conf.device_type == 'gpu':
                    from apex.parallel import DistributedDataParallel as DDP
                    self.model = DDP(self.model)
                    self.head = DDP(self.head)
                    self.head_without_ddp = self.head.module
                else:
                    # NPU DDP 只能修饰一次
                    self.model = DistributedDataParallel(self.model,
                                                         device_ids=[conf.device_id],
                                                         broadcast_buffers=False)
                self.model_without_ddp = self.model.module

        else:
            self.threshold = conf.threshold

    def save_state(self, conf, extra=None, epoch=0):
        save_path = conf.model_path

        ckpt = {
            'model': self.model_without_ddp.state_dict(),
            'head': self.head_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'config': conf,
        }
        torch.save(
            ckpt,
            save_path / ('model_{}_epoch:{}_{:.5f}.pth'.format(get_time(), epoch, extra))
        )

    def load_state_dict(self, weights_path, is_finetune=False):
        ckpt = torch.load(weights_path, map_location='cpu')
        if 'model' in ckpt:
            self.model_without_ddp.load_state_dict(ckpt['model'])
            if not is_finetune:
                self.head_without_ddp.load_state_dict(ckpt['head'])
                self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            self.model_without_ddp.load_state_dict(ckpt)

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        print('========================')
        print(f'{db_name}_accuracy: \t{accuracy}')
        print(f'{db_name}_best_threshold: \t{best_threshold}')
        print('========================')

    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch).detach().cpu().numpy()
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).detach().cpu().numpy()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch).detach().cpu().numpy()
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).detach().cpu().numpy()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def train(self, conf, epochs):
        self.model.train()
        self.head.train()
        running_loss = 0.
        for e in range(self.start_epoch, epochs):
            if conf.is_master_node:
                print('epoch {} started'.format(e))

            # 调节学习率
            if e in self.milestones:
                self.schedule_lr(conf.is_master_node)

            # 分布式训练下，保证每块device所获得的数据不同
            if conf.distributed:
                print("distributed set epoch: ", e)
                self.loader.sampler.set_epoch(e)

            # 同于记录每一步花费时间
            res = []
            for index, (imgs, labels) in enumerate(self.loader):
                start = time.time()

                # train
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = self.loss_func(thetas, labels)

                # 是否使用apex半精度优化进行反向传播
                if conf.use_amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    running_loss += scaled_loss.item()
                else:
                    loss.backward()
                    running_loss += loss.item()

                self.optimizer.step()

                # record cost time
                end = time.time()
                res.append(end - start)

                # 输出日志loss
                if conf.is_master_node and index % self.board_loss_every == 0 and index != 0:
                    loss_board = running_loss / self.board_loss_every
                    now_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    print(f'time:[{now_time}] \tstep: [{index}] \ttrain_loss: {loss_board}')
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                if index == conf.max_iter:
                    break
                self.step += 1

            # eval
            if conf.is_master_node:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                self.model.train()
                self.save_state(conf, extra=accuracy, epoch=e)

            if conf.is_master_node:
                # 去除前5步热身阶段带来的影响
                time_sum = sum(res[5:])
                print('***************************************')
                print("fps: %f" % (((len(res) - 5) * self.loader.batch_size) / time_sum))
                print('***************************************')

    def schedule_lr(self, is_master_node):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        if is_master_node:
            print(self.optimizer)

    def infer(self, conf, faces, target_embs, tta=False):
        """
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        """
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
