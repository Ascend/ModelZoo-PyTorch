# coding=utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import ast
import functools
import glob
import logging
import os
import re
from unittest import mock

import yaml
from easydict import EasyDict

import modelarts_utils
import main as train_main
from pthtar2onx import convert


_CACHE_TRAIN_DATA_URL = "/cache/train_data_url"
_CACHE_TEST_DATA_URL = "/cache/test_data_url"
_CACHE_TRAIN_OUT_URL = "/cache/train_url"


def parse_args():
    parser = argparse.ArgumentParser(description="train crnn")
    # 模型输出目录
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    # 数据集目录
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')
    parser.add_argument('--test_data_url', type=str, default='',
                        help='the test data')
    parser.add_argument('--onnx', default=True, type=ast.literal_eval,
                        help="convert pth model to onnx")
    args = parser.parse_args()

    return args


def load_args_from_config_file(params_file_path):
    with open(params_file_path, 'r') as params_file:
        params_config = yaml.load(params_file)
        params_config = EasyDict(params_config)
    print("Load params config from %s success: %r" %
          (params_file_path, params_config))
    return params_config


def mock_main_parse_arg():
    params_file_path = os.path.join(modelarts_utils.get_cur_path(__file__),
                                    os.path.pardir, "LMDB_config.yaml")
    config = load_args_from_config_file(params_file_path)

    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.DATASET.TRAIN_ROOT = _CACHE_TRAIN_DATA_URL
    config.DATASET.TEST_ROOT = _CACHE_TEST_DATA_URL

    return config


@mock.patch.object(train_main, 'parse_arg', mock_main_parse_arg)
def train():
    train_main.main()


def cmp_acc(x, y):
    """sort by acc and epoch."""
    pattern = r'checkpoint_(?P<epoch>\d+)_acc_(?P<acc>[.0-9]).pth'
    m_x = re.search(pattern, x)
    m_y = re.search(pattern, y)
    if m_x is None or m_y is None:
        print(f"paste x {x} or y {y} failed.")
        return (x > y) - (x < y)

    delta = float(m_x['acc']) - float(m_y['acc'])
    if delta > 0:
        val = 1
    elif delta < 0:
        val = -1
    else:
        val = 0
    return val if val != 0 else int(m_x['epoch']) - int(m_y['epoch'])


def convert_pth_to_onnx():
    pth_pattern = os.path.join(_CACHE_TRAIN_OUT_URL,
                               'output', '*', 'checkpoints', 'checkpoint*.pth')
    pth_list = glob.glob(pth_pattern)
    if not pth_list:
        print (f"can't find pth {pth_pattern}")
        return
    pth_list.sort(key=functools.cmp_to_key(cmp_acc))
    pth = pth_list[-1]
    onnx_path = pth + '.onnx'
    convert(pth, onnx_path)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = parse_args()
    print("Training setting args:", args)

    try:
        import moxing as mox
        print('import moxing success.')

        os.makedirs(_CACHE_TRAIN_DATA_URL, exist_ok=True)
        mox.file.copy_parallel(args.data_url, _CACHE_TRAIN_DATA_URL)

        os.makedirs(_CACHE_TEST_DATA_URL, exist_ok=True)
        mox.file.copy_parallel(args.test_data_url, _CACHE_TEST_DATA_URL)

        # 改变工作目录，用于模型保存
        os.makedirs(_CACHE_TRAIN_OUT_URL, exist_ok=True)
        os.chdir(_CACHE_TRAIN_OUT_URL)

        train()
        if args.onnx:
            print("convert pth to onnx")
            convert_pth_to_onnx()

        mox.file.copy_parallel(_CACHE_TRAIN_OUT_URL, args.train_url)
    except ModuleNotFoundError:
        print('import moxing failed')
        train()


if __name__ == '__main__':
    main()
