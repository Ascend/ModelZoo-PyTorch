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


"""
Post-processing script
"""
import json
import torch
from deepspeech_pytorch.utils import load_decoder
from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.validation import WordErrorRate, CharErrorRate
from hydra.utils import to_absolute_path

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Deepspeech')
parser.add_argument('--out_path', default='./result', type=str, help='infer out path')
parser.add_argument('--info_path', default='./data/an4_dataset/test', type=str, help='infer info path')
parser.add_argument('--label_file', default='./labels.json')
args = parser.parse_args()


def read_dataset(out_path):
    """
    Read the output file
    """
    out_files = os.listdir(out_path)
    # print(out_files)
    data_all = []
    for j in range(len(out_files)//2):
        with open(out_path + '/' + 'data' + str(j + 1) + '_0.txt', 'r') as file:
            data_read = file.read()
            data_line = str(data_read).split(' ')
            data_line.pop(-1)
            data_list = []
            for i in range(311):
                data_list.append(list(map(float, data_line[29 * i: 29 * (i + 1)])))
            data_all.append(data_list)

    # float_list = list(map(float, data_all))
    out_dataset = torch.Tensor(data_all)
    return out_dataset


def read_sizes(info_path):
    """
    Read the sizes file
    """
    with open(info_path + '/sizes/sizes.txt', 'r') as sizes_file:
        sizes_read = sizes_file.read()
        sizes_line = str(sizes_read).split(' ')
        sizes_line.pop(-1)
        sizes_list = list(map(int, sizes_line))
        sizes_list = torch.Tensor(sizes_list).int()
    return sizes_list


def read_targets(info_path):
    """
    Read the targets file
    """
    with open(info_path + '/targets.txt', 'r') as targets_file:
        targets_read = targets_file.read()
        targets_line = str(targets_read).split(' ')
        targets_line.pop(-1)
        targets_list = list(map(int, targets_line))
        targets_list = torch.Tensor(targets_list).int()
    # print(targets_list)
    return targets_list


def read_target_sizes(info_path):
    """
    Read the target sizes file
    """
    with open(info_path + '/target_sizes.txt', 'r') as target_sizes_file:
        target_sizes_read = target_sizes_file.read()
        target_sizes_line = str(target_sizes_read).split(' ')
        target_sizes_line.pop(-1)
        target_sizes_list = list(map(int, target_sizes_line))
        target_sizes_list = torch.Tensor(target_sizes_list).int()
        # print(target_sizes_list)
    return target_sizes_list


if __name__ == '__main__':
    out_dataset = read_dataset(args.out_path)
    out_sizes = read_sizes(args.info_path)
    targets = read_targets(args.info_path)
    target_sizes = read_target_sizes(args.info_path)
    out_sizes = (out_sizes / 2).int()
    device = torch.device("cuda" if EvalConfig.model.cuda else "cpu")
    with open(to_absolute_path(args.label_file)) as label_file:
        labels = json.load(label_file)

    decoder = load_decoder(
        labels=labels,
        cfg=EvalConfig.lm
    )

    target_decoder = GreedyDecoder(
        labels=labels,
        blank_index=labels.index('_')
    )
    # print("模型输出的数据")
    # print(out_dataset)
    # o,_ = target_decoder.decode(out_dataset, out_sizes)
    # print("结果",o)
    wer = WordErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    cer = CharErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    wer.update(
        preds=out_dataset,
        preds_sizes=out_sizes,
        targets=targets,
        target_sizes=target_sizes
    )
    cer.update(
        preds=out_dataset,
        preds_sizes=out_sizes,
        targets=targets,
        target_sizes=target_sizes
    )
    wer = wer.compute()
    cer = cer.compute()
    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
