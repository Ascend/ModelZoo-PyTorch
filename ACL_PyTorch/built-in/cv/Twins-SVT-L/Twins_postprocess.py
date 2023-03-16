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

import argparse
import os
import time

import numpy as np
import torch
from timm.utils import accuracy
from torchvision import datasets, transforms
from tqdm import tqdm


def get_target(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_val = datasets.ImageFolder(image_path, transform=transform)
    # 进行顺序采样
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=1, drop_last=False)
    i = 0
    targets = dict()
    pbar = tqdm(data_loader_val, ncols=100)
    pbar.set_description('Reading Targets： ')
    for _, target in pbar:
        img_path_list = data_loader_val.dataset.imgs[i][0].split(os.path.sep)
        img_name = f"{img_path_list[-1].rstrip('.JPEG')}_0.txt"
        targets[img_name] = target
        i += 1
    return targets


def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        try:
            lines = f.readlines()[0].strip(' \n').split(' ')
        except Exception as e:
            print(str(e))
    target = [list(map(np.float32, lines))]
    boxes = torch.tensor(target)
    return boxes


def get_pred(pred_path):
    """
    {'img1':tensor, 'img2':tensor}
    """
    pred_results = os.listdir(pred_path)
    i = 0
    pred = dict()
    pbar = tqdm(pred_results, ncols=100)
    pbar.set_description('Reading Predictions： ')

    for pred_result in pbar:
        if not pred_result.endswith('.txt'):
            continue
        _boxes = read_pred_file(os.path.join(pred_path, pred_result))
        imgname = os.path.basename(pred_result).split('-')
        imgname = imgname[1] if len(imgname) > 1 else imgname[0]
        i += 1
        pred[imgname] = _boxes
    return pred


def eval_result(loss_acc_list):
    count = len(loss_acc_list)
    loss_total = acc1_total = acc5_total = 0

    for loss, acc1, acc5 in loss_acc_list:
        loss_total += loss
        acc1_total += acc1
        acc5_total += acc5
    return loss_total / count, acc1_total / count, acc5_total / count


def evaluation(image_path, pred_path):
    criterion = torch.nn.CrossEntropyLoss()
    targets = get_target(image_path)
    preds = get_pred(pred_path)

    loss_acc_list = []
    i = 0
    pbar = tqdm(targets.items(), ncols=100)
    pbar.set_description('Start evaluation： ')

    for file_, target in pbar:
        output = preds.get(file_)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_acc_list.append((loss, acc1, acc5))
        i += 1

    loss, acc1, acc5 = eval_result(loss_acc_list)
    print(f"Accuracy of the network on the {len(targets)} val images: {acc1:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default="result/dumpOutput_device0/")
    parser.add_argument('--input_path', default='bin_path')
    args = parser.parse_args()
    evaluation(args.input_path, args.pred)
