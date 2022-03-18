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
"""Script for multi-gpu training."""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.utils.data
from tqdm import tqdm

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord

import random
from apex import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import time
import torch.npu
from apex.optimizers import NpuFusedAdam
worker_num = 16

if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train(opt, train_loader, m, criterion, optimizer,gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)

    train_loader = tqdm(train_loader, dynamic_ncols=True)
    end = time.time()
    for i, (inps, labels, label_masks, _, bboxes) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.npu().requires_grad_() for inp in inps]
        else:
            inps = inps.npu().requires_grad_()
        labels = labels.npu()
        label_masks = label_masks.npu()
        output = m(inps)

        if cfg.LOSS.get('TYPE') == 'MSELoss':
            loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))
            acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))
        else:
            loss = criterion(output, labels, label_masks)
            acc = calc_integral_accuracy(output, labels, label_masks, output_3d=False, norm_type=norm_type)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        # add apex code

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()   
        # loss.backward()
        optimizer.step()
        opt.trainIters += 1
        batch_time.update(time.time() - end)
        end = time.time()
        # TQDM
        train_loader.set_description(
            'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                loss=loss_logger.avg,
                acc=acc_logger.avg)
        )
    if opt.world_size==1 or(opt.world_size>1 and gpu==0):
        Batch_size=32
        if batch_time.avg > 0:
            logger.info('#####npu * FPS@all {:.3f}, TIME@all {:.3f}'.format(opt.world_size * Batch_size / batch_time.avg, batch_time.avg))
    train_loader.close()

    return loss_logger.avg, acc_logger.avg


def validate(gpu, m, opt, heatmap_to_coord, batch_size=20):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=worker_num, drop_last=False)
    kpt_json = []
    eval_joints = det_dataset.EVAL_JOINTS

    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds in tqdm(det_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.npu() for inp in inps]
        else:
            inps = inps.npu()
        output = m(inps)

        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = crop_bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i][det_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i, 0].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(scores[i] + np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open(os.path.join(opt.work_dir, 'test_kpt_{}.json'.format(gpu)), 'w') as fid:
        json.dump(kpt_json, fid)
    ANNfile = './exp/person_keypoints_val2017_{}.json'.format(gpu)
    # res = evaluate_mAP(os.path.join(opt.work_dir, 'test_kpt_{}.json'.format(gpu)), ann_type='keypoints', ann_file=ANNfile)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'test_kpt_{}.json'.format(gpu)), ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return res


def validate_gt(gpu, m, opt, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=worker_num, drop_last=False)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, labels, label_masks, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.npu() for inp in inps]
        else:
            inps = inps.npu()
        output = m(inps)
        
        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i][gt_val_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)
            
            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open(os.path.join(opt.work_dir, 'test_gt_kpt_{}.json'.format(gpu)), 'w') as fid:
        json.dump(kpt_json, fid)
    ANNfile = './exp/person_keypoints_val2017_{}.json'.format(gpu)
    # res = evaluate_mAP(os.path.join(opt.work_dir, 'test_gt_kpt_{}.json'.format(gpu)), ann_type='keypoints', ann_file=ANNfile)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'test_gt_kpt_{}.json'.format(gpu)), ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return res


def main():
    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    torch.manual_seed(0)

    os.environ['MASTER_ADDR'] = opt.addr
    os.environ['MASTER_PORT'] = '59629'

    mp.spawn(parallel_train, nprocs=opt.world_size, args=(cfg, opt)) 

def parallel_train(gpu, cfg, opt):
    torch.npu.set_device('npu:{}'.format(gpu)) 

    dist.init_process_group(
        backend='hccl',
        world_size=opt.world_size,
        rank=gpu
    ) 
    # Model Initialize
    m = preset_model(cfg)
    m = m.npu()

    criterion = builder.build_loss(cfg.LOSS).npu()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = NpuFusedAdam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True)
    if opt.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=gpu
        )
    else:
        train_sampler=None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), num_workers = worker_num,sampler=train_sampler)

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    opt.trainIters = 0

    m, optimizer= amp.initialize(m, optimizer, opt_level="O2",combine_grad=True)
    m = nn.parallel.DistributedDataParallel(m, device_ids=[gpu], broadcast_buffers=False)

    # checkpoint = torch.load('amp_checkpoint.pt')
    # m.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # amp.load_state_dict(checkpoint['amp'])
    # print("load success")
    # with torch.no_grad():
    #     gt_AP = validate_gt(gpu, m, opt, cfg, heatmap_to_coord)
    #     rcnn_AP = validate(gpu, m, opt, heatmap_to_coord)
    #     if opt.world_size==1 or gpu==0:
    #         logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP} | rcnn mAP: {rcnn_AP} #####')  
    
    # exit()

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        if opt.world_size > 1:
            train_sampler.set_epoch(i)
        opt.epoch = i
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        # loss, miou = train(opt, train_loader, m, criterion, optimizer, writer)
        loss, miou = train(opt, train_loader, m, criterion, optimizer,gpu)

        logger.epochInfo('Train', opt.epoch, loss, miou)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
                    # Prediction Test
            with torch.no_grad():
                # gt_AP = validate_gt(m.module, opt, cfg, heatmap_to_coord)
                # rcnn_AP = validate(m.module, opt, heatmap_to_coord)
                gt_AP = validate_gt(gpu, m, opt, cfg, heatmap_to_coord)
                #rcnn_AP = validate(gpu, m, opt, heatmap_to_coord)
                if opt.world_size==1 or gpu==0:
                    logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP}  #####')  
                    torch.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
                    logger.info(f'##### ./exp/{opt.exp_id}-{cfg.FILE_NAME}/model_{opt.epoch}.pth saved!#####')  
        # if (i + 1) % opt.snapshot == 0:
        #     # Save checkpoint
        #     #torch.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
        #     if opt.world_size==1 or gpu==0:
        #         torch.save(m.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))

        #     # Prediction Test
        #     with torch.no_grad():
        #         # gt_AP = validate_gt(m.module, opt, cfg, heatmap_to_coord)
        #         # rcnn_AP = validate(m.module, opt, heatmap_to_coord)
        #         if opt.world_size==1 or gpu==0:
        #             gt_AP = validate_gt(m, opt, cfg, heatmap_to_coord)
        #             rcnn_AP = validate(m, opt, heatmap_to_coord)
        #             logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP} | rcnn mAP: {rcnn_AP} #####')

        # # Time to add DPG
        # if i == cfg.TRAIN.DPG_MILESTONE:
        #     # torch.save(m.module.state_dict(), './exp/{}-{}/final.pth'.format(opt.exp_id, cfg.FILE_NAME))
        #     if opt.world_size==1 or gpu==0:
        #         torch.save(m.state_dict(), './exp/{}-{}/final.pth'.format(opt.exp_id, cfg.FILE_NAME))

        #     # Adjust learning rate
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = cfg.TRAIN.LR
        #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
        #     # Reset dataset
        #     train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, dpg=True)
        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=opt.nThreads)

    #torch.save(m.module.state_dict(), './exp/{}-{}/final_DPG.pth'.format(opt.exp_id, cfg.FILE_NAME))
    if opt.world_size==1 or gpu==0:
        torch.save(m.module.state_dict(), './exp/{}-{}/final_DPG.pth'.format(opt.exp_id, cfg.FILE_NAME))



def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == "__main__":
    main()
