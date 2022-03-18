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
import time
import shutil
import argparse
from tqdm import tqdm
from opts import opts
from logger import Logger
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

# 清空终端
os.system('clear')


def eval(opt, res_data_save_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    # 创建目录
    if os.path.exists(res_data_save_path):
        shutil.rmtree(res_data_save_path)
    os.makedirs(res_data_save_path)

    print('\n[INFO] Infering ...')
    results = {}
    num_iters = len(dataset)
    total_infer_time = 0
    total_infer_num = 0
    for ind in tqdm(range(num_iters)):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        start_time = time.perf_counter()
        ret = detector.run(img_path)
        end_time = time.perf_counter()
        total_infer_time += end_time - start_time
        total_infer_num += 1

        results[img_id] = ret['results']
    print('\n[INFO] Infer done!')

    print('\n[INFO] Calculate accuracy ...')
    dataset.run_eval(results, res_data_save_path)

    # 推理时间
    print('\n[INFO] Time:')
    msg = 'total infer num: ' + str(total_infer_num) + '\n' + \
          'total infer time(ms): ' + str(total_infer_time * 1000) + '\n' + \
          'average infer time(ms): ' + str(total_infer_time * 1000 / total_infer_num) + '\n'
    print(msg)


if __name__ == '__main__':
    '''
    Using Example:

    python pth_eval.py --res_data_save_path=./pth_result
    '''

    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_data_save_path', required=True)

    opt = parser.parse_args()
    res_data_save_path = opt.res_data_save_path

    # 创建并解析opt
    opt = opts().init('--task ctdet --exp_id coco_dla --not_prefetch_test '
                      '--load_model ../models/ctdet_coco_dla_2x.pth'.split(' '))

    # 处理数据
    eval(opt, res_data_save_path)
