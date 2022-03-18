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
import os
import numpy as np
import json
import sys
import argparse

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument('--c', type=int, default=1,
                    help="convert result to json")
parser.add_argument('--i', type=str, default='om_res',
                    help="input om res dir path")
parser.add_argument('--o', type=str, default='om_res.json',
                    help="output om res json file")
parser.add_argument('--t', type=str, default='UCF101bin_batch_info.json',
                    help="target json file path")

class EvalMetric(object):

    def __init__(self, name, **kwargs):
        self.name = str(name)
        self.reset()

    def update(self, preds, labels):
        raise NotImplementedError()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, format(self.sum_metric / self.num_inst,'0.4f'))


class Accuracy(EvalMetric):
    """Computes accuracy classification score.
    """

    def __init__(self, name='accuracy', topk=1):
        super(Accuracy, self).__init__(name)
        self.topk = topk

    def update(self, preds, labels):
        preds = [torch.tensor(preds)]
        labels = [torch.tensor(labels)]

        for pred, label in zip(preds, labels):
            assert self.topk <= pred.shape[1], \
                "topk({}) should no larger than the pred dim({})".format(self.topk, pred.shape[1])
            _, pred_topk = pred.topk(self.topk, 1, True, True)

            pred_topk = pred_topk.t()
            correct = pred_topk.eq(label.view(1, -1).expand_as(pred_topk))
            self.sum_metric += float(correct.contiguous().view(-1).float().sum(0, keepdim=True).numpy())
            self.num_inst += label.shape[0]

def postProcess(result_path, class_num, output_path):
    class_num = int(class_num)
    datatmp = os.listdir(result_path)[0]
    bin_list = os.listdir(os.path.join(result_path,datatmp))
    outputs = []
    labels = []
    for bin_dir in bin_list:
        bin_path = os.path.join(result_path,datatmp, bin_dir)
        name = bin_dir.split('.')[0]
        label = '_'.join(name.split('_')[2:4])
        labels.append(label)
        output = np.loadtxt(bin_path).reshape(-1, class_num)
        outputs.append(output.tolist())
    res = dict(zip(labels, outputs))
    with open(output_path, 'w') as f:
        json.dump(res, f)
    return res


def eval(res, target_file, output):
    with open(target_file, 'r') as f:
        targets = json.load(f)
    if targets is None:
        print('targets can not load : error')
        return
    error_num = 0
    error_list = []
    acc1 = Accuracy(name='acc-1', topk=1)
    acc5 = Accuracy(name='acc-5', topk=5)
    for label, value in res.items():
        if label in targets:
            acc5.update(preds=value, labels=targets[label])
            acc1.update(preds=value, labels=targets[label])
        else:
            error_num += 1
            error_list.append(label)
    if error_num > 0:
        print('error_num', error_num)
        print(error_list)
    print(acc1.get())
    print(acc5.get())
    result = {'acc1':acc1.get(),'acc5':acc5.get()}
    with open(output_path+'_result.txt','w') as f:
        json.dump(result,f)


if __name__ == '__main__':

    args = parser.parse_args()

    input_path = args.i
    target_file = args.t
    output_path = args.o
    print(args.c)
    ss = output_path.split('.')
    if len(ss)==1:
        output_path = output_path+'.json'
    elif ss[1] != 'json':
        print('file should be json')


    if args.c == 1:
        ress = postProcess(input_path, 101, output_path)
    

    with open(output_path, 'r') as f:
        ress = json.load(f)

    eval(ress, target_file, output_path)
