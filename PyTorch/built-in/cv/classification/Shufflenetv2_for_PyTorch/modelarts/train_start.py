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
import ast
import glob
import importlib
import logging
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import yaml

import modelarts_utils
from pthtar2onnx import convert


_MODULE_8P_MAIN_MED = importlib.import_module("8p_main_med")
_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"


def update_argparse_args(args, params):
    args.__dict__.update(params)


def load_args_from_config_file(args):

    params_file_path = os.path.join(
        modelarts_utils.get_cur_path(__file__), "params.yml")
    with open(params_file_path, 'r') as params_file:
        params_config = yaml.load(params_file)
        print("Load params config from %s success: %r" %
              (params_file_path, params_config))
    # 更新参数
    update_argparse_args(args, params_config)

    return args


def get_special_args_for_modelarts(args):
    data = _CACHE_DATA_URL
    resume = args.resume
    if resume:
        # 预训练模型放在相对于当前脚本的目录下
        resume = os.path.join(modelarts_utils.get_cur_path(__file__),
                              resume.lstrip('/'))
    return {
        'data': data,
        'resume': resume,
    }


def parse_args():
    parser = _MODULE_8P_MAIN_MED.parser
    # 模型输出目录
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    # 数据集目录
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')
    parser.add_argument('--onnx', default=True, type=ast.literal_eval,
                        help="convert pth model to onnx")

    # 参数优先级：命令行 > 配置文件 > 默认参数
    default_args = parser.parse_args([])
    config_args = load_args_from_config_file(default_args)
    args = parser.parse_args(namespace=config_args)

    update_argparse_args(args, get_special_args_for_modelarts(args))

    return args


def train(args):
    print("===============trans args=================")
    print(args)
    print("===============trans args=================")

    os.environ['LOCAL_DEVICE_ID'] = str(0)
    print("+++++++++++++++++++++++++++LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    os.environ['MASTER_ADDR'] = args.addr  # '10.136.181.51'
    os.environ['MASTER_PORT'] = '29688'

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.device_list != '':
        ngpus_per_node = len(args.device_list.split(','))
    elif args.device_num != -1:
        ngpus_per_node = args.device_num
    elif args.device == 'npu':
        ngpus_per_node = torch.npu.device_count()
    else:
        ngpus_per_node = torch.cuda.device_count()

    main_worker = _MODULE_8P_MAIN_MED.main_worker
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set LOCAL_DEVICE_ID for every proc
        if args.device == 'npu':
            # main_worker(args.gpu, ngpus_per_node, args)
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def convert_pth_to_onnx(args):
    pth_pattern = os.path.join(_CACHE_TRAIN_URL, 'checkpoint.pth.tar')
    pth_list = glob.glob(pth_pattern)
    if not pth_list:
        print (f"can't find pth {pth_pattern}")
        return
    pth = pth_list[0]
    onnx_path = pth + '.onnx'
    convert(pth, onnx_path, args.num_classes)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = parse_args()
    print("Training setting args:", args)

    try:
        import moxing as mox
        print('import moxing success.')

        os.makedirs(_CACHE_DATA_URL, exist_ok=True)
        mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)

        # 改变工作目录，用于模型保存
        os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
        os.chdir(_CACHE_TRAIN_URL)

        train(args)

        if args.onnx:
            print("convert pth to onnx")
            convert_pth_to_onnx(args)

        mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)
    except ModuleNotFoundError:
        print('import moxing failed')
        train(args)


if __name__ == '__main__':
    main()
