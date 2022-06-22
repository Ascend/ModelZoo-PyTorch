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
# limitations under the License

import argparse

from tqdm import tqdm
import torch
from torch import nn
from torch import optim

import apex
from apex import amp

from model import Backbone, Arcface
from utils import separate_bn_paras


def get_data(args):
    x = torch.rand((args.batch, 3, 112, 112), dtype=torch.float32)
    y = torch.randint(2, (args.batch,), dtype=torch.long)
    return x, y


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.backbone = Backbone(num_layers=100, drop_ratio=0.6, mode='ir_se')
        self.head = Arcface(embedding_size=512, classnum=85742)

    def forward(self, images, labels):
        embeddings = self.backbone(images)
        thetas = self.head(embeddings, labels)
        return thetas


def prepare_args():
    parser = argparse.ArgumentParser(description='get prof')
    parser.add_argument("-device", help="device", default='cuda:0', type=str)
    parser.add_argument("-amp", help="use amp", default=True, type=str)
    parser.add_argument("-batch", help="batch size", default=256, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 640.982ms
    args = prepare_args()
    device = torch.device(args.device)
    if 'npu' in args.device:
        torch.npu.set_device(device)
    else:
        torch.cuda.set_device(device)

    # model
    model = NewModel()
    model = model.to(device)
    print('model head create over ')

    # optimizer
    paras_only_bn, paras_wo_bn = separate_bn_paras(model.backbone)
    if 'npu' in args.device and args.amp:
        optimizer = apex.optimizers.NpuFusedSGD([
                    {'params': paras_wo_bn + [model.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=0.001, momentum=0.9)
    else:
        optimizer = optim.SGD([
                    {'params': paras_wo_bn + [model.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=0.001, momentum=0.9)
    print('optimizer create over')

    # loss function
    loss_func = nn.CrossEntropyLoss().to(device)

    # amp setting
    if 'npu' in args.device and args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0, combine_grad=True)
    elif 'cuda' in args.device and args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)

    print('start warm up train')
    # warm up train
    for _ in tqdm(range(5)):
        imgs, labels = get_data(args)
        imgs = imgs.to(device)
        labels = labels.to(device)

        thetas = model(imgs, labels)
        loss = loss_func(thetas, labels)
        optimizer.zero_grad()

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

    print('start get prof')
    # get prof
    if "npu" in args.device:
        k_v = {'use_npu': True}
    else:
        k_v = {'use_cuda': True}
    with torch.autograd.profiler.profile(**k_v) as prof:
        imgs, labels = get_data(args)
        imgs = imgs.to(device)
        labels = labels.to(device)

        thetas = model(imgs, labels)

        loss = loss_func(thetas, labels)
        optimizer.zero_grad()

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

   # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    prof.export_chrome_trace("output.prof")  # "output.prof"为输出文件地址
