"""
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import sys

datadir = sys.argv[1]
batchsize = 128


def info_nce_loss(features):
    """generate picture matrix"""
    labels = torch.cat([torch.arange(batchsize) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    logits = logits / 0.07
    return logits, labels


def accuracy(output, target, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = topk
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def file_tensor(file_num):
    """txt_file data to tensor"""
    filename = "Simclr_prep_" + str(file_num) + "_0.txt"
    filepath = os.path.join(datadir, filename)
    l = np.loadtxt(filepath, dtype=np.float32)
    dim_res = np.array(l)
    return torch.from_numpy(dim_res)


if __name__ == "__main__":
    top1_accuracy = 0
    ran = int(10000 / batchsize)
    with open('precision.txt', 'w') as f:
        for counters in range(ran):
            images = torch.zeros(batchsize * 2, 128)
            for counter in range(batchsize):
                a = random.randint(1, 19999)
                if a % 2 == 0:
                    image = file_tensor(a)
                    images[counter] = image
                    des = file_tensor(a + 1)
                    images[counter + batchsize] = des
                else:
                    image = file_tensor(a - 1)
                    images[counter] = image
                    des = file_tensor(a)
                    images[counter + batchsize] = des
            logit, label = info_nce_loss(images)
            top1 = accuracy(logit, label, topk=1)
            top1_accuracy += top1[0]
        print("accuracy1 = {}".format(top1_accuracy.item() / (counters + 1)))
