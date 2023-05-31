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

# coding=utf-8

import argparse

class ArgumentParser:
    def __init__(self):
        self.__parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def add_config_argument(self):
        self.__parser.add_argument(
            '--config',
            type=str,
            default='lib/config/360CC_config.yaml',
            help='配置文件'
        )

    def add_checkpoint_argument(self):
        self.__parser.add_argument(
            '--checkpoint',
            type=str,
            default='output/checkpoints/mixed_second_finetune_acc_97P7.pth',
            help='权重文件'
        )

    def add_total_image_dir_argument(self):
        self.__parser.add_argument(
            '--total-image-dir',
            type=str,
            default='images/total_images',
            help='所有图片目录，包括训练集和测试集'
        )

    def add_test_image_dir_argument(self):
        self.__parser.add_argument(
            '--test-image-dir',
            type=str,
            default='images/test_images',
            help='测试集图片目录'
        )

    def add_test_label_argument(self):
        self.__parser.add_argument(
            '--test-label',
            type=str,
            default='lib/dataset/txt/test.txt',
            help='测试集标签文件'
        )

    def add_preprocessed_test_label_argument(self):
        self.__parser.add_argument(
            '--preprocessed-test-label',
            type=str,
            default='lib/dataset/txt/preprocessed_test.txt',
            help='预处理后的测试集标签文件'
        )

    def add_predict_dir_argument(self):
        self.__parser.add_argument(
            '--predict-dir',
            type=str,
            default='ais_bench_output/result',
            help='ais_bench 推理结果目录'
        )
    
    def add_is_dym(self):
        self.__parser.add_argument(
            '--is-dym',
            type=bool,
            default=False,
            help='是否使用动态逻辑'
        )

    def parse_args(self):
        return self.__parser.parse_args()
