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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

import torch
import numpy as np

from utils.vis import save_debug_images_multi
from utils.vis import save_debug_3d_images
from utils.vis import save_debug_3d_cubes
from apex import amp

logger = logging.getLogger(__name__)


def train_3d(config, model, optimizer, loader, epoch, output_dir, writer_dict, num_device=0, device=torch.device('cuda:0'), is_master_node=False, use_apex=False, dtype=torch.float):
    batch_time = AverageMeter(name="batch_time")
    data_time = AverageMeter(name="data_time")
    losses = AverageMeter(name="losses")
    losses_2d = AverageMeter(name="losses_2d")
    losses_3d = AverageMeter(name="losses_3d")
    losses_cord = AverageMeter(name="losses_cord")

    model.train()

    if model.module.backbone is not None:
        model.module.backbone.eval()  # Comment out this line if you want to train 2D backbone jointly

    accumulation_steps = 4
    accu_loss_3d = 0

    end = time.time()
    for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):
        data_time.update(time.time() - end)

        if 'panoptic' in config.DATASET.TEST_DATASET:
            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(views=inputs, meta=meta,
                                                                              targets_2d=targets_2d,
                                                                              weights_2d=weights_2d,
                                                                              targets_3d=targets_3d[0])
        elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
            for tmpi in range(len(targets_3d)):
                if isinstance(targets_3d[tmpi], torch.Tensor):
                    targets_3d[tmpi] = targets_3d[tmpi].float()
                    targets_3d[tmpi]=targets_3d[tmpi].to(device)
            for tmpi in range(len(input_heatmap)):
                if isinstance(input_heatmap[tmpi], torch.Tensor):
                    input_heatmap[tmpi] = input_heatmap[tmpi].float()
                    input_heatmap[tmpi]=input_heatmap[tmpi].to(device)
            for e1 in meta:
                for k, v in e1.items():
                    if isinstance(v, torch.Tensor):
                        v=v.float()
                        e1[k]=v.to(device)
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, torch.Tensor):
                                vv=vv.float()
                                e1[k][kk]=vv.to(device)
            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(meta=meta, targets_3d=targets_3d[0],
                                                                                  input_heatmaps=input_heatmap)

        loss_2d = loss_2d.mean()
        loss_3d = loss_3d.mean()
        loss_cord = loss_cord.mean()

        losses_2d.update(loss_2d.item())
        losses_3d.update(loss_3d.item())
        losses_cord.update(loss_cord.item())
        loss = loss_2d + loss_3d + loss_cord
        losses.update(loss.item())

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.npu.memory_allocated(device)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                  'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                  'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                    data_time=data_time, loss=losses, loss_2d=losses_2d, loss_3d=losses_3d,
                    loss_cord=losses_cord, memory=gpu_memory_usage)
            if is_master_node:
                logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_3d', losses_3d.val, global_steps)
            writer.add_scalar('train_loss_cord', losses_cord.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            for k in range(len(inputs)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{:08}_{}'.format(
                    os.path.join(output_dir, 'train'), i, view_name)
                save_debug_images_multi(config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix)
            prefix2 = '{}_{:08}'.format(
                os.path.join(output_dir, 'train'), i)

            save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
            save_debug_3d_images(config, meta[0], pred, prefix2, is_master_node)

    fps = 'BatchSize:{:.1f}, FPS@all {:.3f}, TIME@all {:.3f}'.format(config.TRAIN.BATCH_SIZE, config.TRAIN.BATCH_SIZE * num_device / batch_time.avg,
                                                      batch_time.avg)
    if is_master_node:
        logger.info(fps)


def validate_3d(config, model, loader, output_dir, device=torch.device('cuda:0'), is_master_node=False):
    batch_time = AverageMeter(name="batch_time")
    data_time = AverageMeter(name="data_time")
    model.eval()

    preds = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):
            data_time.update(time.time() - end)
            if 'panoptic' in config.DATASET.TEST_DATASET:
                pred, heatmaps, grid_centers, _, _, _ = model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                              weights_2d=weights_2d, targets_3d=targets_3d[0])
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                for tmpi in range(len(targets_3d)):
                    if isinstance(targets_3d[tmpi], torch.Tensor):
                        targets_3d[tmpi] = targets_3d[tmpi].float()
                        targets_3d[tmpi]=targets_3d[tmpi].to(device)
                for tmpi in range(len(input_heatmap)):
                    if isinstance(input_heatmap[tmpi], torch.Tensor):
                        input_heatmap[tmpi] = input_heatmap[tmpi].float()
                        input_heatmap[tmpi]=input_heatmap[tmpi].to(device)
                for e1 in meta:
                    for k, v in e1.items():
                        if isinstance(v, torch.Tensor):
                            v=v.float()
                            e1[k]=v.to(device)
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                if isinstance(vv, torch.Tensor):
                                    vv=vv.float()
                                    e1[k][kk]=vv.to(device)
                pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, targets_3d=targets_3d[0],
                                                              input_heatmaps=input_heatmap)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.npu.memory_allocated(device)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                if is_master_node:
                    logger.info(msg)

                for k in range(len(inputs)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{:08}_{}'.format(
                        os.path.join(output_dir, 'validation'), i, view_name)
                    save_debug_images_multi(config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix)
                prefix2 = '{}_{:08}'.format(
                    os.path.join(output_dir, 'validation'), i)

                save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
                save_debug_3d_images(config, meta[0], pred, prefix2, is_master_node)

    metric = None
    if 'panoptic' in config.DATASET.TEST_DATASET:
        aps, _, mpjpe, recall = loader.dataset.evaluate(preds)
        msg = 'ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        if is_master_node:
            logger.info(msg)
        metric = np.mean(aps)
    elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
        actor_pcp, avg_pcp, _, recall = loader.dataset.evaluate(preds)
        msg = '     | Actor 1 | Actor 2 | Actor 3 | Average | \n' \
              ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500mm: {recall:.4f}'.format(
                pcp_1=actor_pcp[0]*100, pcp_2=actor_pcp[1]*100, pcp_3=actor_pcp[2]*100, pcp_avg=avg_pcp*100, recall=recall)
        logger.info(msg)
        metric = np.mean(avg_pcp)

    return metric


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', start_count_index=5):
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
            self.batchsize = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.batchsize):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.batchsize)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
