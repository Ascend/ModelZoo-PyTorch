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


import argparse
import os
import os.path as osp
from sys import path
import warnings

import numpy as np
import mmcv
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

from ais_bench.infer.interface import InferSession, MemorySummary
from ais_bench.infer.summary import summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='i3d inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
             ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=1,
        help='batch size')
    parser.add_argument(
        '--device_id', type=int, default=1,
        help='device id')
    parser.add_argument(
        '--model', required=True, type=str,
        help='i3d.om')
    parser.add_argument(
        '--show', type=bool, default=False,
        help='show h2d time and d2h time')
    args = parser.parse_args()

    return args


def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret = {}".format(message, ret))


class I3d():
    def __init__(self, device_id, model) -> None:
        self.device_id = device_id
        self.model = model

    def inference(self, data_loader):
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            input_data = np.array(data['imgs'])
            result = self.i3d_context([input_data])
            result = torch.from_numpy(np.array(result))
            batch_size = result.shape[1]
            result = result.view(result.shape[0], batch_size, -1)
            result = result.float()
            result = F.softmax(result, dim=2).mean(dim=1)
            result = result.numpy()
            results.extend(result)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()

        print('\n')
        s = model.sumary()
        summary.npu_compute_time_list = s.exec_time_list
        summary.h2d_latency_list = MemorySummary.get_H2D_time_list()
        summary.d2h_latency_list = MemorySummary.get_D2H_time_list()
        if args.show:
            summary.report(opt.batch_size, output_prefix=None, display_all_summary=True)
        else:
            summary.report(opt.batch_size, output_prefix=None, display_all_summary=False)
        return results


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    cfg.data.test.test_mode = True

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=args.batch_size,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)
    dataloader_settings = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_settings)

    i3d = I3d(args.device_id, args.model)
    outputs = i3d.inference(data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        if output_config.get('out', None):
            out = output_config['out']
            print(f'\nwriting results to {out}')
            dataset.dump_results(outputs, **output_config)
        if eval_config:
            eval_res = dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
