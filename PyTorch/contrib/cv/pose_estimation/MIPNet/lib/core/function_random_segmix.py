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
def cutout(input_foreground, segmentation_foreground, input_background, segmentation_background):
    C, H, W = input_foreground.size()

    paste_loc_x = np.random.randint(28, 256-28)
    paste_loc_y = np.random.randint(21, 192-21)

    segmentation_background = 1.0*(segmentation_foreground[:, :]==0)

    # ------------------------------
    segmentation_foreground_idx = (segmentation_foreground == 1).nonzero() 
    segmentation_background_idx = (segmentation_background == 1).nonzero() 
    segmentation_context_idx = (segmentation_context == 1).nonzero() 

    # ------------------------------
    canvas = torch.zeros((C, 3*H, 3*W))

    ## we have 3 layers, first the background of A, then foreground of B, then foreground A
    ## copy input foreground pixels

    # ## paste background of A
    origin_x = W; origin_y = H 
    segmentation_idx = segmentation_background_idx
    canvas[:, origin_y+segmentation_idx[:, 0], origin_x+segmentation_idx[:, 1]] = input_foreground[:, segmentation_idx[:, 0], segmentation_idx[:, 1]]

    # ## paste foreground of B, is the top left of B in the canvas
    origin_x = paste_loc_x; origin_y = paste_loc_y 
    segmentation_idx = segmentation_context_idx
    canvas[:, origin_y+segmentation_idx[:, 0], origin_x+segmentation_idx[:, 1]] = input_context[:, segmentation_idx[:, 0], segmentation_idx[:, 1]]

    ## paste foreground of A
    origin_x = W; origin_y = H 
    segmentation_idx = segmentation_foreground_idx
    canvas[:, origin_y+segmentation_idx[:, 0], origin_x+segmentation_idx[:, 1]] = input_foreground[:, segmentation_idx[:, 0], segmentation_idx[:, 1]]

    # ------------------------------
    origin_x = W; origin_y = H 
    input1 = canvas[:, origin_y:origin_y+H, origin_x:origin_x+W]

    # ------------------------------
    origin_x = paste_loc_x; origin_y = paste_loc_y 
    input2 = canvas[:, origin_y:origin_y+H, origin_x:origin_x+W]
    
    return input1, input2, canvas

# ---------------------------------------------------------------------------------
def random_segmix_data(input, target, target_weight, segmentation):

    batch_size = input.size(0)
    index = torch.randperm(batch_size).cuda()

    ## past location is in range [200, 150] to ensure overlap > 0.1
    ## x range = [28, 256-28] as 28 = (256-200)/2
    ## y range = [21, 192-21] as 21 = (192-150)/2


    # -------- create new tensors----------
    input_A = input.clone();   ## B x 3 x 256 x 192     
    target_A = target.clone();  ## B x 17 x 64 x 48        
    target_weight_A = target_weight.clone().view(-1, 17) ## B x 17
    segmentation_A = segmentation.clone() ##B x 256 x 192

    input_B = input[index].clone();   ## B x 3 x 256 x 192     
    target_B = target[index].clone();  ## B x 17 x 64 x 48        
    target_weight_B = target_weight[index].clone().view(-1, 17) ## B x 17
    segmentation_B = segmentation[index].clone() ##B x 256 x 192


    # ------------------------------------
    segmentation_B_ = segmentation_B.view(-1, 1, input.size(2), input.size(3))
    segmentation_B = segmentation_B_.repeat(1, 3, 1, 1) ## B x 3 x 256 x 192
    not_segmentation_B = 1.0*(segmentation_B[:, :, :, :]==0)

    input_mix = input_A
    input_mix[:, :, bbx1:bbx2, bby1:bby2] = segmentation_B[:, :, bbx1:bbx2, bby1:bby2]*input_B[:, :, bbx1:bbx2, bby1:bby2] +\
                                            not_segmentation_B[:, :, bbx1:bbx2, bby1:bby2]*input_A[:, :, bbx1:bbx2, bby1:bby2]


    segmentation_B = segmentation_B_.repeat(1, 17, 1, 1) ## B x 17 x 256 x 192
    not_segmentation_B = 1.0*(segmentation_B[:, :, :, :]==0)

    target_mix = target_A
    target_mix[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t] = segmentation_B[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t]*target_B[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t] +\
                                                    not_segmentation_B[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t]*target_A[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t]

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



# -----------------------------------------------------------------------------------
def train_random_segmix(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta, segmentation) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, target,target_weight = random_segmix_data(input, target, target_weight, segmentation)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

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
            msg = 'Syn Epoch: [{0}][{1}/{2}]\t' \
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

                prefix = '{}_{}_syn'.format(os.path.join(output_dir, 'train'), i)
                save_debug_images(config, input[:16, [2,1,0], :, :], meta, target[:16], (pred*4)[:16], output[:16],
                                  prefix)
