import argparse
import time
import yaml
import os
import logging
# import tqdm
from pathlib import Path
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import models
import numpy as np

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


def _parse_args():

    parser = argparse.ArgumentParser(description='T2T-ViT Validate.')
    parser.add_argument('--data-dir', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--out-dir', type=str, metavar='PATH', help='path to eval checkpoint')
    parser.add_argument('--gt-path', type=str, metavar='PATH', help='path to groundtruth')
    args = parser.parse_args()

    args.prefetcher = True
    args.distributed = False
    args.device = 'cpu'
    args.world_size = 1
    args.rank = 0
    args.img_size = 224
    args.interpolation = ''
    args.mean = None
    args.std = None
    args.crop_pct = None
    args.channels_last = False
    args.tta = 0
    args.log_interval = 50
    args.local_rank = 0
    args.model_ema = True
    args.model_ema_force_cpu = False
    args.model_ema_decay = 0.99996
    args.seed = 42
    args.batch_size = 1
    args.num_classes = 1000

    return args


def load_val_data(args):

    data_config = resolve_data_config(vars(args), verbose=args.local_rank == 0)
    dataset_eval = Dataset(args.data_dir)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=8,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
    )

    return loader_eval


def pre_process(loader, output_dir, gt_path, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_idx = len(loader) - 1
    labels = []
    num = 0
    # for batch_idx, (input, target) in tqdm.tqdm(enumerate(loader), desc="processing"):
    for batch_idx, (input, target) in enumerate(loader):
        #print(target)
        last_batch = batch_idx == last_idx
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        if num == 0:
            bin_data = input.numpy()
            num += 1
        elif num > 0 and num < 64:
            bin_data = np.concatenate((bin_data,input.numpy()),axis=0)
            num += 1
        elif num == 64:
            print(bin_data.shape)
            num = 1
            save_path = output_dir / f"{batch_idx-1:0>5d}.bin"
            bin_data.tofile(save_path)
            bin_data = input.numpy()
        labels.append(target)

    np.save(gt_path, np.vstack(labels))


def main():
    setup_default_logging()
    args = _parse_args()
    torch.manual_seed(args.seed + args.rank)

    loader = load_val_data(args)
    pre_process(loader, args.out_dir, args.gt_path, args)



if __name__ == '__main__':
    main()
