# coding: utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
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
import glob
import os
import subprocess
import ast
import moxing as mox
from selfexport import pth2onnx


_CACHE_ROOT_URL = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir))
_CACHE_TRAIN_OUT_URL = os.path.join(_CACHE_ROOT_URL, 'output')
_CACHE_TRAIN_DATA_URL = os.path.join(_CACHE_ROOT_URL, 'data')

def parse_args():
    parser = argparse.ArgumentParser(description="train hourglass")
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')
    parser.add_argument('--ckpt_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--total_epochs', type=int, default=1)
    parser.add_argument('--onnx', default=True, type=ast.literal_eval,
                        help="convert pth model to onnx")
    args = parser.parse_args()

    return args


def dos2unix(pattern):
    file_list = glob.glob(pattern)
    try:
        subprocess.run(['dos2unix', *file_list], shell=False, check=True)
    except Exception as exp:
        print(f"run dos2unix failed, {exp}")
        raise exp


def find_latest_pth_file(pth_save_dir):
    pth_pattern = os.path.join(pth_save_dir, '*.pth')
    pth_list = glob.glob(pth_pattern)
    if not pth_list:
        print(f"Cant't found pth in {pth_save_dir}")
        exit()
    pth_list.sort(key=os.path.getmtime)
    print("==================== %s will be exported to .onnx model next! ====================" % pth_list[-1])
    return os.path.join(pth_list[-1])


def main():
    args = parse_args()
    print("Training setting args:", args)
    os.makedirs(_CACHE_TRAIN_OUT_URL, mode=0o750, exist_ok=True)
    os.makedirs(_CACHE_TRAIN_DATA_URL, mode=0o750, exist_ok=True)
    mox.file.copy_parallel(args.data_url, _CACHE_TRAIN_DATA_URL)

    root_dir = _CACHE_ROOT_URL
    os.chdir(root_dir)
    dos2unix("modelarts/*.sh")

    cfg_script = os.path.join(root_dir, 
        'mmpose-master/configs/top_down/hourglass/mpii/hourglass52_mpii_384x384.py')

    train_script = os.path.join(root_dir, 'modelarts/train_1p.sh')
    if os.path.islink(train_script):
        raise Exception(f"{train_script} shouldn't be a link ")
    os.chmod(train_script, 0o500)

    cmd = [
        train_script,
        f'--ckpt_interval={args.ckpt_interval}',
        f'--eval_interval={args.eval_interval}',
        f'--lr={args.lr}',
        f'--warmup_iters={args.warmup_iters}',
        f'--warmup_ratio={args.warmup_ratio}',
        f'--total_epochs={args.total_epochs}',
        f'--work_dir={_CACHE_TRAIN_OUT_URL}', 
        f'--data_root={_CACHE_TRAIN_DATA_URL}',
        f'--config={cfg_script}',
    ]

    try:
        cp = subprocess.run(cmd, shell=False, check=True)
        if cp.returncode != 0:
            print("run train script failed")
            exit(1)

        print("train success")

        if args.onnx:
            latest_pth_file = find_latest_pth_file(_CACHE_TRAIN_OUT_URL)
            pth2onnx(cfg_file=cfg_script,
                    pth_file=latest_pth_file,
                    output_file=os.path.join(_CACHE_TRAIN_OUT_URL, 'hourglass.onnx'),
                    input_shape=[32, 3, 384, 384])

    except Exception as exp:
        print(f"run {cmd} failed, {exp}")
        raise exp
    finally:
        mox.file.copy_parallel(_CACHE_TRAIN_OUT_URL, args.train_url)


if __name__ == '__main__':
    main()
