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
from utils.utils import get_network_grad_flow

# from utils.vis import save_pretty_debug_images as save_debug_images

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
def train_lambda_012(config, train_loader, model, criterion_lambda, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, print_prefix=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model_grads = AverageMeter()
    diversity_losses = AverageMeter()
    pose_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target_a, target_weight_a, meta_a, target_b, target_weight_b, meta_b, target_c, target_weight_c, meta_c) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        B, C, H, W = input.shape

        ##--- 0s and 1s--------
        lambda_val = 0  ##binary dim0: 0, dim1: 0
        lambda_vec_zero = torch.zeros(B, 2).cuda()

        lambda_val = 1 ##binary dim0: 1, dim1: 0
        lambda_vec_one = torch.zeros(B, 2).cuda()
        lambda_vec_one[:, 0] += 1


        lambda_val = 2 ##binary dim0: 0, dim1: 1
        lambda_vec_two = torch.zeros(B, 2).cuda()
        lambda_vec_two[:, 1] += 1

        # lambda_val = torch.cat([torch.zeros(B), torch.zeros(B)+1, torch.zeros(B)+2], dim=0) ### 3B x 2
        # lambda_vec = torch.cat([lambda_vec_zero, lambda_vec_one, lambda_vec_two], dim=0) ### 3B x 2

        # --------------duplicate-----------------------------
        # num_candidates = 3
        # input = torch.cat([input]*num_candidates, dim=0)
        
        # target_a = torch.cat([target_a]*num_candidates, dim=0)
        # target_weight_a = torch.cat([target_weight_a]*num_candidates, dim=0)
        # meta_a['joints'] = torch.cat([meta_a['joints']]*num_candidates, dim=0)
        # meta_a['joints_vis'] = torch.cat([meta_a['joints_vis']]*num_candidates, dim=0)
        
        # target_b = torch.cat([target_b]*num_candidates, dim=0)
        # target_weight_b = torch.cat([target_weight_b]*num_candidates, dim=0)
        # meta_b['joints'] = torch.cat([meta_b['joints']]*num_candidates, dim=0)
        # meta_b['joints_vis'] = torch.cat([meta_b['joints_vis']]*num_candidates, dim=0)

        # target_c = torch.cat([target_c]*num_candidates, dim=0)
        # target_weight_c = torch.cat([target_weight_c]*num_candidates, dim=0)
        # meta_c['joints'] = torch.cat([meta_c['joints']]*num_candidates, dim=0)
        # meta_c['joints_vis'] = torch.cat([meta_c['joints_vis']]*num_candidates, dim=0)

        # # --------------------------------
        # # compute output
        # outputs = model(input, lambda_vec)

        # target_a = target_a.cuda(non_blocking=True)
        # target_weight_a = target_weight_a.cuda(non_blocking=True)

        # target_b = target_b.cuda(non_blocking=True)
        # target_weight_b = target_weight_b.cuda(non_blocking=True)

        # target_c = target_c.cuda(non_blocking=True)
        # target_weight_c = target_weight_c.cuda(non_blocking=True)
        
        # output = outputs

        # start_idx = 0; end_idx = start_idx + B
        # loss_a_lambda = criterion_lambda(output[start_idx:end_idx], target_a, target_weight_a) ##size = B

        # start_idx = B; end_idx = start_idx + B
        # loss_b_lambda = criterion_lambda(output[start_idx:end_idx], target_b, target_weight_b) ##size = B
        
        # start_idx = 2*B; end_idx = start_idx + B
        # loss_c_lambda = criterion_lambda(output[start_idx:end_idx], target_c, target_weight_c) ##size = B

        # pose_loss = loss_a_lambda.mean() + loss_b_lambda.mean() + loss_c_lambda.mean()
        # loss = pose_loss

        # # compute gradient and do update step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # # --------------------------------
        target_a = target_a.cuda(non_blocking=True)
        target_weight_a = target_weight_a.cuda(non_blocking=True)

        target_b = target_b.cuda(non_blocking=True)
        target_weight_b = target_weight_b.cuda(non_blocking=True)

        target_c = target_c.cuda(non_blocking=True)
        target_weight_c = target_weight_c.cuda(non_blocking=True)
       
        # --------------------------------
        # compute output
        outputs_zero = model(input, lambda_vec_zero)
        loss_a_lambda = criterion_lambda(outputs_zero, target_a, target_weight_a) ##size = B
        loss_a = loss_a_lambda.mean()

        optimizer.zero_grad()
        loss_a.backward()
        optimizer.step()

        # --------------------------------
        outputs_one = model(input, lambda_vec_one)
        loss_b_lambda = criterion_lambda(outputs_one, target_b, target_weight_b) ##size = B
        loss_b = loss_b_lambda.mean()

        optimizer.zero_grad()
        loss_b.backward()
        optimizer.step()
       
        # --------------------------------
        outputs_two = model(input, lambda_vec_two)
        loss_c_lambda = criterion_lambda(outputs_two, target_c, target_weight_c) ##size = B
        loss_c = loss_c_lambda.mean()

        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
              
        # --------------------------------
        output = torch.cat([outputs_zero, outputs_one, outputs_two], dim=0)
        loss = loss_a + loss_b + loss_c
        pose_loss = loss
   
        model_grad = get_network_grad_flow(model)
        model_grads.update(model_grad)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        pose_losses.update(pose_loss.item(), input.size(0))

        start_idx = 0; end_idx = start_idx + B        
        _, avg_acc_a, cnt_a, pred_a = accuracy(output[start_idx:end_idx].detach().cpu().numpy(),
                                         target_a.detach().cpu().numpy())
        
        start_idx = B; end_idx = start_idx + B
        _, avg_acc_b, cnt_b, pred_b = accuracy(output[start_idx:end_idx].detach().cpu().numpy(),
                                         target_b.detach().cpu().numpy())

        start_idx = 2*B; end_idx = start_idx + B
        _, avg_acc_c, cnt_c, pred_c = accuracy(output[start_idx:end_idx].detach().cpu().numpy(),
                                         target_c.detach().cpu().numpy())

        acc.update(avg_acc_a, cnt_a)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        msg = 'Epoch: [{0}][{1}/{2}]\t' \
              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
              'Speed {speed:.1f} samples/s\t' \
              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
              'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
              'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
              'model_grad {model_grad.val:.6f} ({model_grad.avg:.6f})\t' \
              'PoseLoss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  speed=input.size(0)/batch_time.val,
                  data_time=data_time, loss=losses, acc=acc,
                  model_grad=model_grads,
                  pose_loss=pose_losses)
        logger.info(msg)

        if i % config.PRINT_FREQ == 0:
            save_size = min(16, B)
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            meta_a['pred_joints_vis'] = torch.ones_like(meta_a['joints_vis'])
            meta_b['pred_joints_vis'] = torch.ones_like(meta_b['joints_vis'])
            meta_c['pred_joints_vis'] = torch.ones_like(meta_c['joints_vis'])

            prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            
            start_idx = 0; end_idx = start_idx + save_size
            save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_a, target_a[:save_size], (pred_a*4)[:save_size], output[start_idx:end_idx], prefix, suffix='a')

            start_idx = B; end_idx = start_idx + save_size
            save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_b, target_b[:save_size], (pred_b*4)[:save_size], output[start_idx:end_idx], prefix, suffix='b')

            start_idx = 2*B; end_idx = start_idx + save_size
            save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_c, target_c[:save_size], (pred_c*4)[:save_size], output[start_idx:end_idx], prefix, suffix='c')


    return

# --------------------------------------------------------------------------------
def train_lambda_0123(config, train_loader, model, criterion_lambda, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, print_prefix=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model_grads = AverageMeter()
    diversity_losses = AverageMeter()
    pose_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target_a, target_weight_a, meta_a, target_b, target_weight_b, meta_b, target_c, target_weight_c, meta_c, target_d, target_weight_d, meta_d) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        B, C, H, W = input.shape

        ##--- 0s and 1s--------
        lambda_val = 0  ##binary dim0: 0, dim1: 0
        lambda_vec_zero = torch.zeros(B, 2).cuda()

        lambda_val = 1 ##binary dim0: 1, dim1: 0
        lambda_vec_one = torch.zeros(B, 2).cuda()
        lambda_vec_one[:, 0] += 1


        lambda_val = 2 ##binary dim0: 0, dim1: 1
        lambda_vec_two = torch.zeros(B, 2).cuda()
        lambda_vec_two[:, 1] += 1

        lambda_val = 3 ##binary dim1: 1, dim1: 1
        lambda_vec_three = torch.zeros(B, 2).cuda()
        lambda_vec_three[:, 0] += 1
        lambda_vec_three[:, 1] += 1

        # # --------------------------------
        target_a = target_a.cuda(non_blocking=True)
        target_weight_a = target_weight_a.cuda(non_blocking=True)

        target_b = target_b.cuda(non_blocking=True)
        target_weight_b = target_weight_b.cuda(non_blocking=True)

        target_c = target_c.cuda(non_blocking=True)
        target_weight_c = target_weight_c.cuda(non_blocking=True)

        target_d = target_d.cuda(non_blocking=True)
        target_weight_d = target_weight_d.cuda(non_blocking=True)
       
        # --------------------------------
        # compute output
        outputs_zero = model(input, lambda_vec_zero)
        loss_a_lambda = criterion_lambda(outputs_zero, target_a, target_weight_a) ##size = B
        loss_a = loss_a_lambda.mean()

        optimizer.zero_grad()
        loss_a.backward()
        optimizer.step()

        # --------------------------------
        outputs_one = model(input, lambda_vec_one)
        loss_b_lambda = criterion_lambda(outputs_one, target_b, target_weight_b) ##size = B
        loss_b = loss_b_lambda.mean()

        optimizer.zero_grad()
        loss_b.backward()
        optimizer.step()
       
        # --------------------------------
        outputs_two = model(input, lambda_vec_two)
        loss_c_lambda = criterion_lambda(outputs_two, target_c, target_weight_c) ##size = B
        loss_c = loss_c_lambda.mean()

        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        
        # --------------------------------
        outputs_three = model(input, lambda_vec_three)
        loss_d_lambda = criterion_lambda(outputs_three, target_d, target_weight_d) ##size = B
        loss_d = loss_d_lambda.mean()

        optimizer.zero_grad()
        loss_d.backward()
        optimizer.step()


        # --------------------------------
        output = torch.cat([outputs_zero, outputs_one, outputs_two, outputs_three], dim=0)
        loss = loss_a + loss_b + loss_c + loss_d
        pose_loss = loss
   
        model_grad = get_network_grad_flow(model)
        model_grads.update(model_grad)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        pose_losses.update(pose_loss.item(), input.size(0))

        start_idx = 0; end_idx = start_idx + B        
        _, avg_acc_a, cnt_a, pred_a = accuracy(output[start_idx:end_idx].detach().cpu().numpy(),
                                         target_a.detach().cpu().numpy())
        
        start_idx = B; end_idx = start_idx + B
        _, avg_acc_b, cnt_b, pred_b = accuracy(output[start_idx:end_idx].detach().cpu().numpy(),
                                         target_b.detach().cpu().numpy())

        start_idx = 2*B; end_idx = start_idx + B
        _, avg_acc_c, cnt_c, pred_c = accuracy(output[start_idx:end_idx].detach().cpu().numpy(),
                                         target_c.detach().cpu().numpy())

        start_idx = 3*B; end_idx = start_idx + B
        _, avg_acc_d, cnt_d, pred_d = accuracy(output[start_idx:end_idx].detach().cpu().numpy(),
                                         target_d.detach().cpu().numpy())

        acc.update(avg_acc_a, cnt_a)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        msg = 'Epoch: [{0}][{1}/{2}]\t' \
              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
              'Speed {speed:.1f} samples/s\t' \
              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
              'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
              'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
              'model_grad {model_grad.val:.6f} ({model_grad.avg:.6f})\t' \
              'PoseLoss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  speed=input.size(0)/batch_time.val,
                  data_time=data_time, loss=losses, acc=acc,
                  model_grad=model_grads,
                  pose_loss=pose_losses)
        logger.info(msg)

        if i % config.PRINT_FREQ == 0:
            save_size = min(16, B)
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            meta_a['pred_joints_vis'] = torch.ones_like(meta_a['joints_vis'])
            meta_b['pred_joints_vis'] = torch.ones_like(meta_b['joints_vis'])
            meta_c['pred_joints_vis'] = torch.ones_like(meta_c['joints_vis'])
            meta_d['pred_joints_vis'] = torch.ones_like(meta_d['joints_vis'])

            prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            
            start_idx = 0; end_idx = start_idx + save_size
            save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_a, target_a[:save_size], (pred_a*4)[:save_size], output[start_idx:end_idx], prefix, suffix='a')

            start_idx = B; end_idx = start_idx + save_size
            save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_b, target_b[:save_size], (pred_b*4)[:save_size], output[start_idx:end_idx], prefix, suffix='b')

            start_idx = 2*B; end_idx = start_idx + save_size
            save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_c, target_c[:save_size], (pred_c*4)[:save_size], output[start_idx:end_idx], prefix, suffix='c')

            start_idx = 3*B; end_idx = start_idx + save_size
            save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_d, target_d[:save_size], (pred_d*4)[:save_size], output[start_idx:end_idx], prefix, suffix='d')

    return
# --------------------------------------------------------------------------------
# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
