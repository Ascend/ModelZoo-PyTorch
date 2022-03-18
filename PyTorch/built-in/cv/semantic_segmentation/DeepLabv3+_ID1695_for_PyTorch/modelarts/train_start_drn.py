# coding: utf-8
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

import argparse
import ast
import glob
import logging
import os
import subprocess

import moxing as mox

import modelarts_utils
from pth2onnx import convert


_CACHE_TRAIN_OUT_URL = os.path.realpath(
    os.path.join(modelarts_utils.get_cur_path(__file__), os.path.pardir))
_CACHE_TRAIN_DATA_URL = os.path.join(_CACHE_TRAIN_OUT_URL, 'data')


def parse_args():
    parser = argparse.ArgumentParser(description="train crnn")
    # Model output directory
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    # Dataset directory
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')
    parser.add_argument('--test_data_url', type=str, default='',
                        help='the test data')
    parser.add_argument('--backbone_dir', type=str, default='',
                        help='backbone dir')
    parser.add_argument('--onnx', default=True, type=ast.literal_eval,
                        help="convert pth model to onnx")

    parser.add_argument('--train_epochs', type=int, default=100,
                        help='train epochs')
    # checking point
    parser.add_argument('--backbone', type=str, default='drn',
                        help='chose the backbone name [resnet, drn]')
    parser.add_argument('--checkname', type=str, default='deeplab-drn',
                        help='set checkname')
    parser.add_argument('--ckpt_name', type=str, default='drn_d_54-0e0534ff.pth',
                        help='set backbone checkpoint name')
    parser.add_argument('--resume', type=str, default='',
                        help='put the path to resuming file if needed')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='dataset num classes')

    return parser.parse_args()


def convert_pth_to_onnx(dataset, num_classes, backbone, checkname):
    pth_pattern = os.path.join(_CACHE_TRAIN_OUT_URL,
                               f'run/{dataset}/{checkname}/',
                               'model_best.pth.tar')
    pth_list = glob.glob(pth_pattern)
    if not pth_list:
        print (f"can't find pth {pth_pattern}")
        return
    pth = pth_list[-1]
    onnx_path = pth + '.onnx'
    convert(pth, onnx_path, num_classes, backbone=backbone)
    print(f"convert {pth} ot {onnx_path} success.")


def dos2unix(pattern):
    file_list = glob.glob(pattern)
    try:
        subprocess.run(['dos2unix', *file_list], shell=False, check=True)
    except Exception as exp:
        print(f"run dos2unix failed, {exp}")
        raise exp


def complete_data_path(r_path):
    """
    Args:
        r_path: relative path

    Returns:

    """
    return os.path.join(_CACHE_TRAIN_DATA_URL, r_path) if r_path else ''


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = parse_args()
    print("Training setting args:", args)
    os.makedirs(_CACHE_TRAIN_DATA_URL, mode=0o750, exist_ok=True)
    mox.file.copy_parallel(args.data_url, _CACHE_TRAIN_DATA_URL)


    # Change the working directory for model saving
    root_dir = _CACHE_TRAIN_OUT_URL
    os.chdir(root_dir)
    dos2unix("test/*.sh")

    train_script = os.path.join(root_dir, 'test/train_full_1p_drn.sh')
    if os.path.islink(train_script):
        raise Exception(f"{train_script} shouldn't be a link ")
    os.chmod(train_script, 0o500)
    cmd = [
        train_script,
        f'--train_epochs={args.train_epochs}',
        f'--dataset={args.dataset}',
        f'--data_path={_CACHE_TRAIN_DATA_URL}',
        f'--more_path1={complete_data_path(args.backbone_dir)}',
        f'--num_classes={args.num_classes}',
        f'--checkname={args.checkname}',
        f'--backbone={args.backbone}',
        f'--ckpt_name={args.ckpt_name}',
        f'--resume={complete_data_path(args.resume)}',
    ]
    try:
        cp = subprocess.run(cmd, shell=False, check=True)
        if cp.returncode != 0:
            print("run train script failed")
            exit(1)

        print("train success")

        if args.onnx:
            print("begin convert pth to onnx")
            convert_pth_to_onnx(args.dataset, args.num_classes, args.backbone, args.checkname)
    except Exception as exp:
        print(f"run {cmd} failed, {exp}")
        raise exp
    finally:
        model_dir = os.path.join(root_dir, 'run')
        log_dir = os.path.join(root_dir, 'test/output')
        print("-----------------------------------")
        print("_CACHE_TRAIN_DATA_URL: ", _CACHE_TRAIN_DATA_URL)
        print("_CACHE_TRAIN_OUT_URL: ", _CACHE_TRAIN_OUT_URL)
        print("resume: ", complete_data_path(args.resume))
        print("model_dir: ", model_dir)
        print("log_dir: ", log_dir)
        print("-----------------------------------")
        mox.file.copy_parallel(model_dir, args.train_url)
        mox.file.copy_parallel(log_dir, args.train_url)


if __name__ == '__main__':
    main()
