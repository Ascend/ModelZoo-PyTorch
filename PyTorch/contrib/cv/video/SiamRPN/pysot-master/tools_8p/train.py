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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np
import apex

try:
    from apex import amp
except:
    print('no apex')

import torch
if torch.__version__>= '1.8':
    import torch_npu
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients, \
    average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
parser.add_argument('--is_performance', action='store_true', default=False,
                    help='test performance or not test')
parser.add_argument('--max_step', type=int, default=2000,
                    help='stop in max step')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None

    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=get_world_size(),
                                           rank=get_rank())

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=False,
                              sampler=train_sampler)

    return train_loader


def build_opt_lr(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    trainable_params = []

    trainable_params += [{'params': model.backbone.parameters(),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rpn_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def train(train_loader, model, optimizer, lr_scheduler):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
                    cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.P8SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.P8SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()

    for idx, data in enumerate(train_loader):

        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                    {'epoch': epoch,
                     'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict()},
                    cfg.TRAIN.P8SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')

                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    for layer in cfg.BACKBONE.TRAIN_LAYERS:
                        for param in getattr(model.module.backbone, layer).parameters():
                            param.requires_grad = True
                        for m in getattr(model.module.backbone, layer).modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.train()
                else:
                    for layer in cfg.BACKBONE.TRAIN_LAYERS:
                        for param in getattr(model.backbone, layer).parameters():
                            param.requires_grad = True
                        for m in getattr(model.backbone, layer).modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.train()

                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch + 1))

        # tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:

            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch + 1, pg['lr']))

        data_time = average_reduce(time.time() - end)

        outputs = model(data)

        loss = outputs['total_loss']

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            reduce_gradients(model)

            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)

            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.data.item())

        average_meter.update(**batch_info)

        if rank == 0:

            if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:

                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                    epoch + 1, (idx + 1) % num_per_epoch,
                    num_per_epoch, cur_lr)
                avgtime = batch_info['batch_time'] + batch_info['data_time']
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                            getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                            getattr(average_meter, k))
                logger.info(info)
                print_speed(idx + 1 + start_epoch * num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
                print('FPS', (28 * 8 / avgtime.item()))
        end = time.time()
        
        if args.is_performance:
            if idx == args.max_step:
                exit()


def main():
    os.environ['RANK'] = str(args.local_rank)

    rank, world_size = dist_init()

    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:

        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().npu().train()

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model)

    # resume training

    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    dist_model, optimizer = amp.initialize(model, optimizer, opt_level="O1",
                                           loss_scale=32)

    dist_model = torch.nn.parallel.DistributedDataParallel(dist_model,
                                                           device_ids=[rank])
    # start training

    train(train_loader, dist_model, optimizer, lr_scheduler)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
