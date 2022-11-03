# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from core.inference import get_max_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.function import AverageMeter


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
def cutmix_data(input, target, target_weight, alpha=1.0, cutmix_prob=1.0):
    ''' Returns mixed inputs, pairs of targets, and lambda'''

    r = np.random.rand(1)
    if alpha > 0 and r < cutmix_prob:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = -1
        return input, target, target_weight, None, lam

    batch_size = input.size()[0]
    index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    bbx1_t, bby1_t, bbx2_t, bby2_t = bbx1 // 4, bby1 // 4, bbx2 // 4, bby2 // 4

    # -------- create new tensors----------
    input_A = input.clone();        target_A = target.clone();          target_weight_A = target_weight.clone().view(-1, 17)
    input_B = input[index].clone(); target_B = target[index].clone();   target_weight_B = target_weight[index].clone().view(-1, 17)

    input_mix = input_A
    input_mix[:, :, bbx1:bbx2, bby1:bby2] =  input_B[:, :, bbx1:bbx2, bby1:bby2]

    target_mix = target_A
    target_mix[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t] = target_B[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t]


    # -----------------------------------------------------
    ##update to mask keypoints inside bb
    keypoints_A, max_vals_A = get_max_preds(target_A.clone().cpu().numpy())
    keypoints_B, max_vals_B = get_max_preds(target_B.clone().cpu().numpy())


    kps_A_inside_bbox = (   (keypoints_A[:, :, 0] >= bbx1_t) * \
                            (keypoints_A[:, :, 0] <= bbx2_t) * \
                            (keypoints_A[:, :, 1] >= bby1_t) * \
                            (keypoints_A[:, :, 1] <= bby2_t)
                        )

    kps_A_outside_bbox = torch.tensor((~kps_A_inside_bbox)*1.0)
    target_weight_mix_A = (kps_A_outside_bbox * target_weight_A).view(-1, 17, 1)


    kps_B_inside_bbox = (   (keypoints_B[:, :, 0] >= bbx1_t) * \
                            (keypoints_B[:, :, 0] <= bbx2_t) * \
                            (keypoints_B[:, :, 1] >= bby1_t) * \
                            (keypoints_B[:, :, 1] <= bby2_t) * 1.0
                        )

    kps_B_inside_bbox = torch.tensor(kps_B_inside_bbox*1.0)
    target_weight_mix_B = (kps_B_inside_bbox * target_weight_B).view(-1, 17, 1)


    # -----------------------------------------------------
    # reweight based on bounding box size.
    lam = 1 - ((bbx2-bbx1) * (bby2-bby1)) / (input.size()[-1] * input.size()[-2])


    # -----------------------------------------------------
    return input_mix, target_mix, target_weight_mix_A, target_weight_mix_B, lam

# ---------------------------------------------------------------------------------
def cutmix_data_no_target_weight_update(input, target, target_weight, alpha=1.0, cutmix_prob=1.0):
    ''' Returns mixed inputs, pairs of targets, and lambda'''

    r = np.random.rand(1)
    if alpha > 0 and r < cutmix_prob:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = -1
        return input, target, target_weight, None, lam

    batch_size = input.size()[0]
    index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    bbx1_t, bby1_t, bbx2_t, bby2_t = bbx1 // 4, bby1 // 4, bbx2 // 4, bby2 // 4

    # -------- create new tensors----------
    input_A = input.clone();        target_A = target.clone();          target_weight_A = target_weight.clone().view(-1, 17)
    input_B = input[index].clone(); target_B = target[index].clone();   target_weight_B = target_weight[index].clone().view(-1, 17)

    input_mix = input_A
    input_mix[:, :, bbx1:bbx2, bby1:bby2] =  input_B[:, :, bbx1:bbx2, bby1:bby2]

    target_mix = target_A
    target_mix[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t] = target_B[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t]

    target_weight_mix_A = target_weight_A.view(-1, 17, 1)
    target_weight_mix_B = target_weight_B.view(-1, 17, 1)

    # -----------------------------------------------------
    # reweight based on bounding box size.
    lam = 1 - ((bbx2-bbx1) * (bby2-bby1)) / (input.size()[-1] * input.size()[-2])


    # -----------------------------------------------------
    return input_mix, target_mix, target_weight_mix_A, target_weight_mix_B, lam

# ---------------------------------------------------------------------------------


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt( 1 - lam )
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx-cut_w//2, 0, W)
    bby1 = np.clip(cy-cut_h//2, 0, H)
    bbx2 = np.clip(cx+cut_w//2, 0, W)
    bby2 = np.clip(cy+cut_h//2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_criterion(criterion, pred, target, tweighta, tweightb, lam):
    return lam * criterion(pred, target, tweighta) + (1-lam) * criterion(pred, target, tweightb)

def train_cutmix(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        m_input, m_target, m_tweia, m_tweib, lam = cutmix_data(input, target,
                target_weight, alpha=1.0, cutmix_prob=1.0)

        # m_input, m_target, m_tweia, m_tweib, lam = cutmix_data_no_target_weight_update(input, target,
        #         target_weight, alpha=1.0, cutmix_prob=1.0)

        # compute output
        outputs = model(m_input)

        m_target = m_target.cuda(non_blocking=True)
        m_tweia = m_tweia.cuda(non_blocking=True)
        if m_tweib is not None:
            m_tweib = m_tweib.cuda(non_blocking=True)

        if isinstance(outputs, list) and m_tweib is not None:
            # loss = criterion(outputs[0], target, target_weight)
            loss = cutmix_criterion(criterion, outputs[0], m_target, m_tweia, m_tweib, lam)
            for output in outputs[1:]:
                # loss += criterion(output, target, target_weight)
                loss += cutmix_criterion(criterion, output, m_target, m_tweia, m_tweib, lam)
        elif not isinstance(outputs, list) and m_tweib is not None:
            output = outputs
            # loss = criterion(output, target, target_weight)
            loss = cutmix_criterion(criterion, output, m_target, m_tweia, m_tweib, lam)
        elif not isinstance(outputs, list) and m_tweib is None:
            output = outputs
            loss = criterion(output, m_target, m_tweia)
        else:
            loss = criterion(outputs[0], m_target, m_tweia)
            for output in outputs[1:]:
                loss += criterion(output, m_target, m_tweia)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                save_debug_images(config, m_input[:, [2,1,0], :, :], meta, m_target, pred*4, output,
                              prefix)
