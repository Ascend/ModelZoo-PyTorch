# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------

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

import os
import json
import socket
import logging
import argparse

import torch
import torch.nn.parallel
import torch.distributed as dist

import dataset
from network.symbol_builder import get_symbol

parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='Kinetics', choices=['Kinetics', 'ucf101'],
                    help="path to dataset")
parser.add_argument('--clip-length', default=8,
                    help="define the length of each input sample.")
parser.add_argument('--train-frame-interval', type=int, default=8,
                    help="define the sampling interval between frames.")
parser.add_argument('--val-frame-interval', type=int, default=8,
                    help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./exps/models",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='RESNET50_3D_GCN_X5',
                    choices=['RESNET50_3D_GCN_X5', 'RESNET101_3D_GCN_X5'],
                    help="chose the base network")
# optimization
parser.add_argument('--pretrained', type=bool, default=True,
                    help="load default pretrained model.")
parser.add_argument('--fine-tune', type=bool, default=False,
                    help="resume training and then fine tune the classifier")
parser.add_argument('--precise-bn', type=bool, default=True,
                    help="try to refine batchnorm layers at the end of each training epoch.")
parser.add_argument('--resume-epoch', type=int, default=-1,
                    help="resume train")
parser.add_argument('--batch-size', type=int, default=64,
                    help="batch size")
parser.add_argument('--lr-base', type=float, default=0.05,
                    help="learning rate")
parser.add_argument('--lr-steps', type=list, default=[int(24*1e4*x) for x in [45,65,85]],
                    help="number of samples to pass before changing learning rate") # 1e6 million
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help="reduce the learning with factor")
parser.add_argument('--save-frequency', type=float, default=1,
                    help="save once after N epochs")
parser.add_argument('--end-epoch', type=int, default=10000,
                    help="maxmium number of training epoch")
parser.add_argument('--random-seed', type=int, default=1,
                    help='random seed (default: 1)')
# distributed training
parser.add_argument('--backend', default='nccl', type=str,
                    help='Name of the backend to use')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://HOSTNAME:23455', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--distributed', default='no', type=str)
parser.add_argument('--local_rank', default=-1, type=int)

parser.add_argument('--apex', default='no', type=str)
parser.add_argument('--apex_level', default='O2', type=str)
parser.add_argument('--loss_scale', default=128.0, type=float)

parser.add_argument('--prof', default='no', type=str)


def autofill(args):
    # customized
    if not args.task_name:
        args.task_name = os.path.basename(os.getcwd())
    if not args.log_file:
        if os.path.exists("./exps/logs"):
            if args.distributed:
                args.log_file = "./exps/logs/{}.log".format(args.task_name)
            else:
                args.log_file = "./exps/logs/{}.log".format(args.task_name)
        else:
            args.log_file = ".{}.log".format(args.task_name)
    # fixed
    args.model_prefix = os.path.join(args.model_dir, args.task_name)
    print("===============args.model_prefix===============")
    print(args.model_prefix)
    return args

def set_logger(args, log_file='', debug_mode=False):

    if log_file:
        if not os.path.exists("./"+os.path.dirname(log_file)):
            os.makedirs("./"+os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                format='%(asctime)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)

if __name__ == "__main__":

    # set args
    args = parser.parse_args()
    args.distributed = True if args.distributed == 'yes' else False
    args = autofill(args)

    # set apex
    args.apex = True if args.apex == 'yes' else False
    args.loss_scale = args.loss_scale if args.loss_scale > 0 else None

    # set prof
    args.prof = True if args.prof == 'yes' else False

    set_logger(args, log_file=args.log_file, debug_mode=args.debug_mode)
    logging.info("==============================================分界线===================================================")
    logging.info("Using pytorch {} ({})".format(torch.__version__, torch.__path__))
    logging.info("Start training with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    assert torch.npu.is_available(), "NPU is not available"
    torch.manual_seed(args.random_seed)
    torch.npu.manual_seed(args.random_seed)

    # distributed setting
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "21000"

    # distributed training
    if args.distributed:
        rank = args.local_rank
        dist_url = args.dist_url
        torch.npu.set_device(rank)
        args.device = torch.device("npu", args.local_rank)

        logging.info("Distributed Training (rank = {}), world_size = {}, backend = `{}', host-url = `{}'".format(
                     rank, args.world_size, args.backend, dist_url))
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='21000')
        dist.init_process_group(backend='hccl', init_method=dist_init_method, world_size=args.world_size, rank=args.local_rank)
        logging.info("Distributed setting end!")
    args.master_node = (not args.distributed) or (torch.distributed.is_initialized and torch.distributed.get_rank() == 0)
    logging.info(f"node {torch.distributed.get_rank() if args.distributed else 0} is master node: {args.master_node}.")

    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)

    # creat model with all parameters initialized
    assert (not args.fine_tune or not args.resume_epoch < 0), \
            "args: `resume_epoch' must be defined for fine tuning"
    net, input_conf = get_symbol(name=args.network,
                     pretrained=args.pretrained if args.resume_epoch < 0 else None,
                     print_net=False, # True if args.distributed else False,
                     **dataset_cfg)

    # training
    kwargs = {}
    kwargs.update(dataset_cfg)
    kwargs.update({'input_conf': input_conf})
    kwargs.update(vars(args))

    if args.master_node:
        logging.info("============kwargs===========")
        logging.info(kwargs)

    from train_model import train_model
    train_model(sym_net=net, args=args, **kwargs)
