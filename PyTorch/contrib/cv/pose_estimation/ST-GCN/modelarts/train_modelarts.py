#!/usr/bin/env python
# Copyright 2020 Huawei Technologies Co., Ltd
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

import moxing as mox
import argparse
from cmath import phase
import os
from net.onnx_net.st_gcn import Model
import torch
import torch.onnx
from torchlight import import_class
import torch.multiprocessing as mp


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval('dict({})'.format(values))  # pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert(model_path):
    checkpoint = torch.load(os.path.join(model_path, "best_model_1p.pt"), map_location='cpu')
    model = Model(in_channels=3,
                  num_class=400,
                  edge_importance_weighting=True,
                  graph_args={'layout': "openpose",
                              'strategy': "spatial"}
                  )
    model.load_state_dict(checkpoint)
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(64, 3, 150, 18, 2)
    torch.onnx.export(model, dummy_input, os.path.join(model_path, "stgcn_npu.onnx"), input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # Cache data set directory
    CACHE_DATA_URL = '/cache/data_url/'
    # Cache output directory
    CACHE_TRAIN_URL = '/cache/train_url/'

    # Data set directory
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')
    # Model output directory
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')

    parser.add_argument('--weights', default=None,
                            help='the weights for network initialization')

    parser.add_argument('-w', '--work_dir', default='./work_dir/recognition/kinetics_skeleton/ST_GCN',
                            help='the work folder for storing results')

    parser.add_argument('-c', '--config', default='config/st_gcn/kinetics-skeleton/train.yaml',
                        help='path to the configuration file')

    # processor
    parser.add_argument('--num_epoch', type=int, default=1,
                        help='stop training in which epoch')
    parser.add_argument('--use_gpu_npu', type=str,
                        default="npu", help='use GPU or NPU')
    parser.add_argument('--device', type=int, default=0, nargs='+',
                        help='the indexes of GPUs for training or testing')
    parser.add_argument('--amp', type=str2bool,
                        default=True, help='use amp or not')

    # feeder
    parser.add_argument('--num_worker', type=int, default=4,
                        help='the number of worker per gpu for data loader')
    parser.add_argument('--train_feeder_args', action=DictAction,
                        default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', action=DictAction,
                        default=dict(), help='the arguments of data loader for test')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='training batch size')
    parser.add_argument('--test_batch_size', type=int,
                        default=64, help='test batch size')

                    
    # read arguments
    args = parser.parse_args()
    root_path = CACHE_DATA_URL
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    mox.file.copy_parallel(args.data_url, root_path)

    argv = []
    # build argv from args
    for arg in vars(args):
        if arg == "data_url" or arg == "train_url":
            continue
        if arg == "train_feeder_args" or arg == "test_feeder_args":
            dc = getattr(args, arg)
            if "data_path" in dc:
                dc["data_path"] = root_path + dc["data_path"]
            if "label_path" in dc:
                dc["label_path"] = root_path + dc["label_path"]
        if arg == "weights":
            if "test.yaml" in args.config :
                argv.append("--"+arg)
                argv.append(root_path + str(getattr(args, arg)))
            continue

        
        argv.append("--"+arg)
        argv.append(str(getattr(args, arg)))
    # start
    Processor = import_class(
        'processor.recognition.REC_Processor')

    p = Processor(argv)
    devices = [p.arg.device] if isinstance(
        p.arg.device, int) else list(p.arg.device)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '59629'
    if len(devices) > 1 or "gpu" in p.arg.use_gpu_npu:
        mp.spawn(p.parallel_train, nprocs=len(devices))
    else:
        p.parallel_train(p.arg.device[0])

    if "train.yaml" in args.config:
        convert(args.work_dir)
    mox.file.copy_parallel(args.work_dir, args.train_url)

