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
import random
import argparse
from pathlib import Path

import numpy as np
import torch
if torch.__version__ >= "1.8.1":
    import torch_npu
import torch.distributed as dist

from config import get_config
from Learner import face_learner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_distributed_mode(conf, args):
    addr, port = args.dist_url.split(':')
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    if 'RANK_SIZE' in os.environ:
        args.rank_size = int(os.environ['RANK_SIZE'])
        args.rank = args.dist_rank * args.rank_size + args.device_id
        args.world_size = args.gpus * args.rank_size

        conf.batch_size = int(args.batch_size / args.rank_size)
        conf.world_size = args.world_size
        conf.rank = args.rank
    else:
        raise RuntimeError("init_distributed_mode failed.")

    print(f'init distributed: device id : {args.device_id} \
            rank id: {args.rank}, \
            world_size: {args.world_size}, \
            dist_rank: {args.dist_rank}')
    torch.distributed.init_process_group(backend=args.backend, init_method="env://",
                                         world_size=args.world_size, rank=args.rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=100, type=int)
    parser.add_argument("--data_mode", help="use which database, [vgg, ms1m, emore, concat]", default='emore', type=str)
    parser.add_argument("--eval_data_mode", help="eval dataset", default='lfw', type=str)
    parser.add_argument("--data_path", help="data dir", default='./data/faces_emore', type=str)
    parser.add_argument("--max_iter", help="max_iter", default=1000, type=int)
    parser.add_argument("--start_epoch", help="train epoch", default=0, type=int)
    parser.add_argument("--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("--seed", help="train seed", default=2021, type=int)

    # finetune
    parser.add_argument("--is_finetune", help="if finetune", default=0, type=int)
    parser.add_argument("--resume", help="reload pretrained weights path", default="", type=str)

    # device type gpu or npu
    parser.add_argument("--device_type", help="device type, [gpu, npu]", default="gpu", type=str)

    # distributed
    parser.add_argument("--device_id", help="device_id", default=0, type=int)
    parser.add_argument("--distributed", help="if distributed", default=0, type=int)
    parser.add_argument("--backend", help="", default='nccl', type=str)
    parser.add_argument("--dist_url", help="", default='127.0.0.1:41111', type=str)
    parser.add_argument("--gpus", help="number of gpus per node", default=1, type=int)
    parser.add_argument("--dist_rank", help="node rank for distributed training", default=0, type=int)

    # apex amp
    parser.add_argument("--use_amp", help="if use amp", default=1, type=int)
    parser.add_argument("--opt_level", help="apex amp level, [O1, O2]", default='O2', type=str)
    parser.add_argument("--loss_scale", help="apex amp loss scale, [128.0, None]", default=128.0)

    # init config
    args = parser.parse_args()
    conf = get_config()
    conf.is_master_node = not args.distributed or args.device_id == 0

    # set seed
    if args.seed:
        set_seed(args.seed)

    # model config
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth

    # train config
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.eval_data_mode = args.eval_data_mode
    conf.emore_folder = Path(args.data_path)
    conf.max_iter = args.max_iter
    conf.start_epoch = args.start_epoch
    conf.is_finetune = args.is_finetune

    # distributed or only one device
    conf.distributed = args.distributed
    conf.device_type = args.device_type
    conf.device_id = args.device_id

    if args.device_type == 'gpu':
        conf.device = torch.device(f"cuda:{args.device_id}")
        torch.cuda.set_device(conf.device)
    elif args.device_type == 'npu':
        conf.device = torch.device(f"npu:{args.device_id}")
        torch.npu.set_device(conf.device)
    else:
        raise ValueError('device type error,please choice in ["gpu","npu"]')

    # distributed config
    if conf.distributed:
        init_distributed_mode(conf, args)

    # apex amp config
    conf.use_amp = args.use_amp
    conf.opt_level = args.opt_level
    conf.loss_scale = args.loss_scale

    learner = face_learner(conf)
    # 加载预训练模型
    if args.resume:
        learner.load_state_dict(args.resume, is_finetune=args.is_finetune)

    learner.train(conf, args.epochs)
