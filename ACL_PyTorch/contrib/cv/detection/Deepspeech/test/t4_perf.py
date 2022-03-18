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

import hydra
import os
import torch
from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.utils import load_model

from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.utils import load_decoder

import os
import json
import numpy as np
import argparse

from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_loader import AudioDataLoader
from hydra.utils import to_absolute_path
import time

parser = argparse.ArgumentParser(description='Deepspeech')
# The data file to read
parser.add_argument('--data_file', default='./data/an4_test_manifest.json')
# The location the generated 'bin' file to save
parser.add_argument('--save_path', default='./data/an4_dataset/test')
args = parser.parse_args()



if __name__ == '__main__':

    device = torch.device("cuda" if EvalConfig.model.cuda else "cpu")
    with open(to_absolute_path(DeepSpeechConfig.data.labels_path)) as label_file:
        labels = json.load(label_file)
    # if labels:
    #     print("labels ready")
    data_module = DeepSpeechDataModule(
        labels=labels,
        data_cfg=DeepSpeechConfig.data,
        normalize=True,
        is_distributed=False # DeepSpeechConfig.trainer.gpus > 1
    )
    dataset = data_module._create_dataset(args.data_file)

    data_loader = AudioDataLoader(
        dataset=dataset,
        num_workers=data_module.data_cfg.num_workers,
        batch_size=data_module.data_cfg.batch_size
    )

    inputs, targets, input_percentages, target_sizes = data_loader.collate_fn(data_loader.dataset)

    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    inputs = inputs.to(device)



    device = torch.device("cuda" if EvalConfig.model.cuda else "cpu")
    model = load_model(device=device, model_path='an4_pretrained_v3.ckpt')
    model.eval()
    model = model.to(device)
    print('Finished loading model!')
    s_time = time.time()
    for i in range(5):
        out, output_sizes = model(inputs[:1], input_sizes[:1])
    e_time = time.time()
    t = (e_time - s_time)/5
    print('Finished testing data!')
    print('Run time of each data: ', t)
    print('performance: ', 1/t, 'seq/s')
