#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a classification model."""
import argparse,sys,os
import torch
if torch.__version__ >= "1.8":
    import torch_npu
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
import pycls.datasets.loader as data_loader

from pycls.core.config import cfg

def init_process_group(proc_rank, world_size, device_id, port='29588'):
    """Initializes the default process group."""

    # Initialize the process group
    print('Initialize the process group')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29588'
    torch.distributed.init_process_group(
        backend=cfg.DIST_BACKEND,
        #init_method="tcp://{}:{}".format(cfg.HOST, port),
        world_size=world_size,
        rank=proc_rank,
        #rank=0
    )
    print("init_process_group done")

    # Set the GPU to use
    #torch.cuda.set_device(proc_rank)
    torch.npu.set_device(device_id)
    print('set_device done.', cfg.DIST_BACKEND, world_size, proc_rank)

def main():
    config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()

    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)


if __name__ == "__main__":
    """Load config from command line arguments and set any specified options."""        
    parser = argparse.ArgumentParser(description="Config file options.")
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file", help=help_s, required=True, type=str)
    parser.add_argument("--rank_id", dest="rank_id", default=0, type=int)
    parser.add_argument("--device_id", dest="device_id", default=0, type=int)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)   
    print("===============1================")
    # modelarts modification
    parser.add_argument('--data_url',
                    metavar='DIR',
                    default='/cache/data_url',
                    help='path to dataset')
    parser.add_argument('--train_url',
                    default="/cache/training",
                    type=str,
                    help="setting dir of training output")
    parser.add_argument('--onnx', default=True, action='store_true',
                    help="convert pth model to onnx")
    parser.add_argument('--npu',
                    default=None,
                    type=int,
                    help='NPU id to use.')
    print("===============2================")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    print(args)
    print("args.data_url:",args.data_url)
    print('cur dir:', os.listdir('./'))
    config.merge_from_file(args.cfg_file)
    config._C.merge_from_list(args.opts)
    config.assert_and_infer_cfg()
    cfg.freeze()
    
    init_process_group(proc_rank=args.rank_id, world_size=cfg.NUM_GPUS,  device_id=args.device_id)
    cur_device = torch.npu.current_device()
    print('cur_device: ', cur_device)
    
    trainer.train_model(args)
    trainer.convert_pth_to_onnx(args)
    
