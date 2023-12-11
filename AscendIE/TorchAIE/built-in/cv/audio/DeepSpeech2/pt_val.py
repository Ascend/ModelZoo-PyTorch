# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import json
import copy
import argparse

import torch
import torch_aie
import numpy as np
from torch_aie import _enums
from torch.utils.data import dataloader
from hydra.utils import to_absolute_path

from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset
from model_pt import forward_infer


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def collate_fn(batch):
    """
    data preprocessing
    """
    def func(p):
        """
        data size
        """
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
    return inputs, input_percentages, []


def get_dataloader(opt):
    with open(to_absolute_path(opt.label_file)) as label_file:
        labels = json.load(label_file)

    dataset = SpectrogramDataset(
        audio_conf=DataConfig.spect,
        input_path=opt.data_file,
        labels=labels,
        normalize=True,
        aug_cfg=DataConfig.augmentation
    )
    inputs, input_percentages, _ = collate_fn(dataset)
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    print(inputs[0])
    print(input_sizes.tolist())

    datasets = [[inputs[i], input_sizes[i]] for i in range(len(input_sizes))]
    while len(datasets) % opt.batch_size != 0:
        datasets.append(datasets[-1])
    m = 1
    datasets_orig = copy.deepcopy(datasets)
    while m < opt.multi:
        datasets += datasets_orig
        m += 1

    loader =  InfiniteDataLoader  # only DataLoader allows for attribute updates
    print(opt.batch_size)
    return loader(datasets,
                  batch_size=opt.batch_size,
                  shuffle=False,
                  num_workers=1,
                  sampler=None,
                  pin_memory=True)


def save_tensor_arr_to_file(arr, file_path):
    write_sen = ""
    for l in arr:
        for c in l:
            write_sen += str(c) + " "
        write_sen += "\n"
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(write_sen)

def save_size_to_file(size, file_path):
    write_sen = "" + str(size) + " "
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(write_sen)

def main(opt):
    # load model
    model = torch.jit.load(opt.model)
    batch_size = opt.batch_size
    torch_aie.set_device(opt.device_id)
    if opt.need_compile:
        inputs = []
        inputs.append(torch_aie.Input([opt.batch_size, 1, 161, 621], dtype=torch_aie.dtype.FLOAT))
        inputs.append(torch_aie.Input([opt.batch_size], dtype=torch_aie.dtype.INT32))

        model = torch_aie.compile(
            model,
            inputs=inputs,
            precision_policy=_enums.PrecisionPolicy.FP16,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version='Ascend310P3',
            optimization_level=0)

    dataloader = get_dataloader(opt)
    pred_results = forward_infer(model, dataloader, batch_size, opt.device_id)

    if opt.batch_size == 1 and opt.multi == 1:
        result_path = opt.result_path
        if(os.path.exists(result_path) == False):
                os.makedirs(result_path)
        for index, res in enumerate(pred_results):
            for i in range(batch_size):
                result_fname_0 = 'data' + str(index * batch_size + i + 1) + '_0.txt'
                result_fname_1 = 'data' + str(index * batch_size + i + 1) + '_1.txt'
                res = np.array(res)
                save_tensor_arr_to_file(np.array(res[0][i]), os.path.join(result_path, result_fname_0))
                save_size_to_file(res[1].numpy()[i], os.path.join(result_path, result_fname_1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech2 offline model inference.')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3', help='soc version')
    parser.add_argument('--model', type=str, default="deepspeech_torchscript_torch_aie_bs1.pt", help='ts model path')
    parser.add_argument('--need_compile', action="store_true", help='if the loaded model needs to be compiled or not')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--data_file', default='./deepspeech.pytorch/data/an4_test_manifest.json')
    parser.add_argument('--label_file', default='./deepspeech.pytorch/labels.json')
    parser.add_argument('--result_path', default='result/dumpout')
    parser.add_argument('--multi', type=int, default=1, help='multiples of dataset replication for enough infer loop. if multi != 1, the pred result will not be stored.')
    opt = parser.parse_args()
    main(opt)
