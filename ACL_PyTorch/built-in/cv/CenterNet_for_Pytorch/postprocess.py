# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import _init_paths
import os
from tqdm import tqdm
import argparse
import shutil
import pickle
import glob
from os.path import join
import numpy as np
import torch
from logger import Logger
from datasets.dataset_factory import dataset_factory
from lib.detectors.ctdet import CtdetDetector
from lib.detectors.detector_factory import detector_factory
from lib.opts import opts
from lib.models.utils import flip_tensor
from lib.models.decode import ctdet_decode

# 清空终端
os.system('clear')


class ModelWarper(CtdetDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, output):
        hm = torch.from_numpy(output['hm']).sigmoid_()
        wh = torch.from_numpy(output['wh'])
        reg = torch.from_numpy(output['reg']) if self.opt.reg_offset else None
        if self.opt.flip_test:
            hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
            wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
            reg = reg[0:1] if reg is not None else None
        dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        return dets

    def run(self, output, meta, scale):
        detections = []
        dets = self.process(output)
        dets = self.post_process(dets, meta, scale)
        detections.append(dets)

        results = self.merge_outputs(detections)
        return results


def postprocess(infer_res_save_path, pre_data_save_path, opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    model_warper = ModelWarper(detector.opt)

    # 创建目录
    if os.path.exists(res_data_save_path):
        shutil.rmtree(res_data_save_path)
    os.makedirs(res_data_save_path)

    # 读取文件个数
    bin_file_list = glob.glob(join(infer_res_save_path, '*.bin'))
    bin_file_num = len(bin_file_list) // 3

    # 加载字典
    image_id_dict = pickle.load(open(join(pre_data_save_path, 'image_id_dict.pkl'), 'rb'))
    meta_dict = pickle.load(open(join(pre_data_save_path, 'meta_dict.pkl'), 'rb'))

    # 后处理
    print('\n[INFO] Postprocessing ...')
    results = {}
    for i in tqdm(range(bin_file_num)):
        hm = np.fromfile(join(infer_res_save_path, str(i) + '_3.bin'), dtype=np.float32).reshape(1, 80, 128, 128)
        wh = np.fromfile(join(infer_res_save_path, str(i) + '_1.bin'), dtype=np.float32).reshape(1, 2, 128, 128)
        reg = np.fromfile(join(infer_res_save_path, str(i) + '_2.bin'), dtype=np.float32).reshape(1, 2, 128, 128)

        output = {'hm': hm, "wh": wh, "reg": reg}
        meta = meta_dict[i]
        scale = [1.0]

        result = model_warper.run(output, meta, scale)

        results[image_id_dict[i]] = result
    print('[INFO] Postprocess done!')

    # 计算精度
    print('\n[INFO] Calculate accuracy ...')
    dataset.run_eval(results, res_data_save_path)


if __name__ == '__main__':
    '''
    Using Example:

    python postprocess.py \
        --infer_res_save_path=./result/dumpOutput_device0 \
        --pre_data_save_path=./pre_bin \
        --res_data_save_path=./om_result
    '''

    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_res_save_path', required=True)
    parser.add_argument('--pre_data_save_path', required=True)
    parser.add_argument('--res_data_save_path', required=True)

    opt = parser.parse_args()
    infer_res_save_path = opt.infer_res_save_path
    pre_data_save_path = opt.pre_data_save_path
    res_data_save_path = opt.res_data_save_path

    # 创建并解析opt
    opt = opts().init('--task ctdet --exp_id coco_dla --not_prefetch_test '
                      '--load_model ../models/ctdet_coco_dla_2x.pth'.split(' '))

    # 处理数据
    results = postprocess(infer_res_save_path, pre_data_save_path, opt)
