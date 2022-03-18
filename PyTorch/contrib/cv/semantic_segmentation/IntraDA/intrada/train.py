# Copyright 2020 Huawei Technologies Co., Ltd
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

#--------------------------------------------------------------------
# modified from "ADVENT/advent/scripts/train.py" by Tuan-Hung Vu
#--------------------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import random
import warnings

import numpy as np
import yaml
import torch
from torch.utils import data

from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.gta5 import GTA5DataSet
from advent.dataset.cityscapes import CityscapesDataSet as CityscapesDataSet_hard
from cityscapes import CityscapesDataSet as CityscapesDataSet_easy
from advent.domain_adaptation.config import cfg, cfg_from_file
from train_UDA import train_domain_adaptation


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz_every_iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--device_type', type=str, default='npu')
    parser.add_argument('--device_id', type=int, )
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--performance_log', action='store_true', default=False)
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    # pdb.set_trace()

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    cfg.distributed = args.distributed
    ddp_backend = "hccl" if args.device_type == "npu" else "nccl"

    if cfg.distributed:
        torch.distributed.init_process_group(backend=ddp_backend, world_size=args.world_size, rank=args.rank)
    device = torch.device("{}:{}".format(args.device_type ,args.device_id))
    cfg.device_id = args.device_id
    cfg.rank = args.rank
    cfg.world_size = args.world_size
    cfg.device_type = args.device_type
    cfg.performance_log = args.performance_log
    if args.device_type == 'cuda':
        torch.cuda.set_device(args.device_id)
    elif args.device_type == 'npu':
        torch.npu.set_device(args.device_id)

    cfg.is_master_node = args.world_size == 1 or args.device_id == 0

    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    if cfg.is_master_node:
        print('Using config:')
        pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        if args.device_type == 'cuda':
            torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        elif args.device_type == 'npu':
            torch.npu.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # LOAD SEGMENTATION NET
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')

    # DATALOADERS
    # pdb.set_trace()
    easy_dataset = CityscapesDataSet_easy(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE * args.world_size,
                                 crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                 mean=cfg.TRAIN.IMG_MEAN)
    if cfg.distributed:
        easy_sampler = torch.utils.data.distributed.DistributedSampler(easy_dataset)
    easy_loader = data.DataLoader(easy_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=(not cfg.distributed),
                                    pin_memory=False,
                                    sampler=easy_sampler if cfg.distributed else None,
                                    worker_init_fn=_init_fn)

    # pdb.set_trace()
    hard_dataset = CityscapesDataSet_hard(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET * args.world_size,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)
    if cfg.distributed:
        hard_sampler = torch.utils.data.distributed.DistributedSampler(hard_dataset)
    hard_loader = data.DataLoader(hard_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=(not cfg.distributed),
                                    pin_memory=False,
                                    sampler=hard_sampler if cfg.distributed else None,
                                    worker_init_fn=_init_fn)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # pdb.set_trace()
    train_domain_adaptation(model, easy_loader, hard_loader, device, cfg)


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    main()
