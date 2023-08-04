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

import time
import argparse

import cv2
import numpy as np
import torch

import torch_aie
from torch_aie import _enums

from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.registry import TASK_UTILS
from mmocr.apis.inferencers.base_mmocr_inferencer import BaseMMOCRInferencer


def pre_data(img_path, bs):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 32))
    img = ((img - mean) / std ).astype(np.float32)
    img = img.transpose(2,0,1)

    data = torch.from_numpy(img)
    data = data.expand(bs, *data.shape)
    return data


def init_model(pth, name='abinet', device='cpu'):
    model = BaseMMOCRInferencer(name, pth, device)
    model = model.model
    model.eval()
    return model

def compile_model(model, data):
    trace_model = torch.jit.trace(model, data)
    shape = data.shape
    input_info = [torch_aie.Input(shape)]
    pt_model = torch_aie.compile(
            trace_model,
            inputs=input_info,
            precision_policy=_enums.PrecisionPolicy.FP16,
            soc_version="Ascend310P3"
            )
    return pt_model

def compute_res(model, data):
    return model(data)

def pos_res(prob, is_print):
    data_dict = {'type': 'Dictionary', 
    'dict_file': 'mmocr/dicts/lower_english_digits.txt', 
    'with_start': True, 
    'with_end': True, 
    'same_start_end': True, 
    'with_padding': False, 
    'with_unknown': False
    }

    dictionary = TASK_UTILS.build(data_dict)
    max_value, max_idx = torch.max(prob[0], -1)

    index, score = [], []
    output_index = max_idx.numpy().tolist()
    output_score = max_value.numpy().tolist()
    for char_index, char_score in zip(output_index, output_score):
        if char_index == dictionary.end_idx:
            break
        index.append(char_index)
        score.append(char_score)

    text = dictionary.idx2str(index)
    score = np.mean(score)
    if is_print:
        print(f"predict text is {text}, the score is {score:0.3f}")

def comute_fps(model, data, loop, warmcount):
    loops = loop

    while warmcount:
        _ = model(data)
        warmcount -= 1

    t0 = time.time()
    while loops:
        _ = model(data)
        loops -= 1
    t1 = time.time() - t0

    # loop * bs / t
    print(f'fps: {loop} * {data.shape[0]} / {t1:.3f} of model is {loop * data.shape[0] / t1:.3f} samples/s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pth',
        type=str,
        default='./abinet_20e_st-an_mj_20221005_012617-ead8c139.pth',
        help="path of abinet's pth"
    )
    parser.add_argument(
        '--img',
        type=str,
        default='mmocr/demo/demo_text_recog.jpg',
        help='path of demo img'
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=1,
        help='bs of model, used to compute fps,default is 1'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='device id, default is 0'
    )
    parser.add_argument(
        '--loop',
        type=int,
        default=1000,
        help='loop count of infer, default is 1000'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=0,
        help='loop count of before infer, default is 0'
    )
    parser.add_argument(
        '--print',
        action='store_true',
        help='print result or not, default is True'
    )

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    opts = parse_args()
    torch_aie.set_device(opts.device)

    abinet = init_model(opts.pth)

    if opts.print:
        imgs = pre_data(opts.img, 1)
        pt_abinet = compile_model(abinet, imgs)
        pt_result = compute_res(pt_abinet, imgs)
        pos_res(pt_result, opts.print)

    if opts.warmup:
        imgs_infer = pre_data(opts.img, opts.bs)
        pt_abinet = compile_model(abinet, imgs_infer)
        comute_fps(pt_abinet, imgs_infer, opts.loop, opts.warmup)
            