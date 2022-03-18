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
import argparse
import shutil
import pickle
import os
import cv2
from os.path import join
from logger import Logger
from tqdm import tqdm
from datasets.dataset_factory import dataset_factory
from lib.detectors.ctdet import CtdetDetector
from lib.detectors.detector_factory import detector_factory
from lib.opts import opts

# 清空终端
os.system('clear')


class ModelWarper(CtdetDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def run(self, image_path, image_save_path, meta_save_path):
        image = cv2.imread(image_path)
        scale = self.scales[0]
        images, meta = self.pre_process(image, scale, None)

        # 保存数据
        images = images.numpy()
        images.tofile(image_save_path)

        return meta


def preprocess(opt, pre_data_save_path):
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
    if os.path.exists(pre_data_save_path):
        shutil.rmtree(pre_data_save_path)
    os.makedirs(pre_data_save_path)

    # 处理数据
    info_file_path = join(pre_data_save_path, 'bin_file.info')
    with open(info_file_path, 'wt', encoding='utf-8') as f_info:
        image_id_dict = {}
        meta_dict = {}
        num_iters = len(dataset)
        for i in tqdm(range(num_iters)):
            image_id = dataset.images[i]
            image_info = dataset.coco.loadImgs(ids=[image_id])[0]
            image_path = join(dataset.img_dir, image_info['file_name'])
            image_save_path = join(pre_data_save_path, str(i) + '.bin')
            meta_save_path = join(pre_data_save_path, str(i) + '.pkl')
            meta = model_warper.run(image_path, image_save_path, meta_save_path)
            f_info.write(str(i) + " ./" + str(i) + '.bin 512 512' + '\n')
            image_id_dict[i] = image_id
            meta_dict[i] = meta
    pickle.dump(image_id_dict, open(join(pre_data_save_path, 'image_id_dict.pkl'), 'wb'))
    pickle.dump(meta_dict, open(join(pre_data_save_path, 'meta_dict.pkl'), 'wb'))


if __name__ == '__main__':
    '''
    Using Example:

    python preprocess.py --pre_data_save_path=./pre_bin
    '''

    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_data_save_path', required=True)
    opt = parser.parse_args()
    pre_data_save_path = opt.pre_data_save_path

    # 创建并解析opt
    opt = opts().init('--task ctdet --exp_id coco_dla --not_prefetch_test '
                      '--load_model ../models/ctdet_coco_dla_2x.pth'.split(' '))

    # 处理数据
    print('\n[INFO] Preprocessing ...')
    preprocess(opt, pre_data_save_path)
    print('[INFO] Preprocess done!')
