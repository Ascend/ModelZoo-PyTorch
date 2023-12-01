# Copyright 2023 Huawei Technologies Co., Ltd
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

# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import time
import tqdm
import argparse
import numpy as np
import torch
import torch_npu
import torch_aie
from torch_aie import _enums
import torch.utils.data

from opts_pose import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from datasets.sample.multi_pose import Multiposebatch
from utils.image import get_affine_transform


def compile_aie_model(torch_model, batch_size, aie_model_path):
    accept_size = [batch_size, 3, 800, 800]
    dummy_input = torch.rand(accept_size) / 2
    with torch.no_grad():
        jit_model = torch.jit.trace(torch_model, dummy_input)
    aie_input_spec = [
        torch_aie.Input(shape=accept_size, dtype=torch_aie.dtype.FLOAT16),
    ]
    aie_model = torch_aie.compile(
        jit_model,
        inputs=aie_input_spec,
        precision_policy=_enums.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        require_full_compilation=False,
        allow_tensor_replace_int=False,
        min_block_size=3,
        torch_executed_ops=[],
        soc_version="Ascend310P3",
        optimization_level=0)
    aie_model.save(aie_model_path)
    return aie_model


def preprocess(img, mean, std):
    height, width = img.shape[0:2]
    inp_height = inp_width = 800
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(img, (width, height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float16)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    # 不使用contiguous内存不连续，模型运行时PT插件内部会做，放到前处理可以使用多流掩盖
    images = torch.from_numpy(images).contiguous()
    return images


def main(pt_model_path, data_path, result_path):
    opt = opts().parse()
    device = opt.gpus_str
    torch_npu.npu.set_device(int(device))
    batch_size = opt.batch_size
    print('Using device id = ', device)
    print('Test batch_size = ', batch_size)
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, pt_model_path, None,
                       opt.resume, opt.lr, opt.lr_step)
    model.eval()
    mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    # 性能测试
    with torch.no_grad():
        origin_path = os.getcwd()
        os.chdir(origin_path)
        aie_model_path = 'centerface_aie_bs' + str(batch_size) + '.pt'
        if not os.path.exists(aie_model_path):
            print('The modelpath {} does not exist, compile aie model.'.format(aie_model_path))
            aie_model = compile_aie_model(
                model, batch_size, aie_model_path)
            aie_model = aie_model.npu()
        else:
            print('The modelpath {} exists, load model.'.format(aie_model_path))
            aie_model = torch.jit.load(aie_model_path).npu()
        aie_model.eval()

        # 性能测试
        random_input = torch.randn(
            [batch_size, 3, 800, 800], dtype=torch.half, device=torch_npu.npu.current_device())
        times = []
        for i in range(3):
            pred = aie_model(random_input)
        print('Warm up for 3 times, done.')
        for i in range(100):
            torch.npu.synchronize()
            start = time.time()
            pred = aie_model(random_input)
            torch.npu.synchronize()
            end = time.time()
            times.append(end - start)
        print('Random input performance test, batchSize = {}, QPS = {}.'.format(
            batch_size, 100 * batch_size / sum(times)))

        # 精度测试
        in_files = os.listdir(data_path)
        model_times = []
        documents = []
        batch_inputs = None
        for file in tqdm.tqdm(sorted(in_files)):
            os.chdir(os.path.join(data_path, file))
            cur_path = os.getcwd()
            doc = os.listdir(cur_path)
            for document in doc:
                if document == 'output':
                    break
                if not document.endswith('jpg'):
                    continue
                image = cv2.imread(os.path.join(cur_path, document))
                images = preprocess(image, mean, std).npu()
                documents.append(document)
                if len(documents) == 1:
                    batch_inputs = images
                else:
                    batch_inputs = torch.concat(
                        (batch_inputs, images), dim=0)
                if len(documents) == batch_size:
                    if len(model_times) == 0:
                        for i in range(3):
                            pred = aie_model(batch_inputs)
                        print('Warm up for 3 times, done.')
                    torch.npu.synchronize()
                    start = time.time()
                    pred = aie_model(batch_inputs)
                    torch.npu.synchronize()
                    end = time.time()
                    model_times.append(end - start)
                    pred = pred.cpu()  # 需要搬运回CPU才能print和切片
                    hm = pred[:, 0:1, :, :]
                    wh = pred[:, 1:3, :, :]
                    hm_offset = pred[:, 3:5, :, :]
                    landmarks = pred[:, 5:15, :, :]
                    for j in range(batch_size):
                        doc = documents[j].strip('\n')
                        dir_name = doc.split(
                            '_')[0] + '--' + doc.split('_')[1]
                        result_dir_path = os.path.join(
                            result_path, 'bs' + str(batch_size), dir_name)
                        if not os.path.exists(os.path.join(result_path, 'bs' + str(batch_size))):
                            os.mkdir(os.path.join(result_path, 'bs' + str(batch_size)))
                        if not os.path.exists(result_dir_path):
                            os.mkdir(result_dir_path)
                        hm[j].unsqueeze(0).numpy().tofile(os.path.join(result_dir_path, 
                                                        documents[j].split('.')[0] + '_0.bin'))
                        wh[j].unsqueeze(0).numpy().tofile(os.path.join(result_dir_path, 
                                                        documents[j].split('.')[0] + '_1.bin'))
                        hm_offset[j].unsqueeze(0).numpy().tofile(os.path.join(result_dir_path, 
                                                                documents[j].split('.')[0] + '_2.bin'))
                        landmarks[j].unsqueeze(0).numpy().tofile(os.path.join(result_dir_path, 
                                                                documents[j].split('.')[0] + '_3.bin'))
                    documents.clear()  # 清空元素，进入下一个batchsize
        print('Dataset infer, infer {} images, inference takes {} second.'.format(
            len(model_times) * batch_size, sum(model_times)))
        print('Dataset infer, batchsize = {}, QPS = {}.'.format(batch_size, len(model_times) * batch_size / sum(model_times)))


if __name__ == "__main__":
    pt_model_path = './model_best.pth'
    data_path = '../../WIDER_val/images'
    result_path = './result'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_path = os.path.abspath(data_path)
    result_path = os.path.abspath(result_path)
    main(pt_model_path, data_path, result_path)
