# Copyright 2021 Huawei Technologies Co., Ltd
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
import os
import logging
import json
import torch

from .distributed import get_rank, synchronize
from .logger import setup_logger
from .env import seed_all_rng
from ..config import cfg

def default_setup(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    # if not args.no_cuda and torch.cuda.is_available():
    #     # cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = True
    #     args.device = "cuda"
    # else:
    #     args.distributed = False
    #     args.device = "cpu"
    if args.distributed:

        torch.distributed.init_process_group(backend='hccl',  # init_method=args.dist_url,
                                world_size=args.world_size, rank=args.local_rank)
        local_device = f'npu:{args.local_rank}'
        torch.npu.set_device(local_device)

        synchronize()
    else:
        local_device = 'npu:3'
        torch.npu.set_device(local_device)

    # TODO
    # if args.save_pred:
    #     outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)

    save_dir = cfg.TRAIN.LOG_SAVE_DIR if cfg.PHASE == 'train' else None
    setup_logger("Segmentron", save_dir, get_rank(), filename='{}_{}_{}_{}_log.txt'.format(
        cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP))

    logging.info("Using {} GPUs".format(num_gpus))
    logging.info(args)
    logging.info(json.dumps(cfg, indent=8))

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + get_rank())