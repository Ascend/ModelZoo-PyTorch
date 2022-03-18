# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================


import argparse
import math
import apex
try:
    from apex import amp
except:
    amp = None
import cv2
import numpy as np
import numpy.random as nprnd
import os
import torch
import torch.utils.data
import tqdm
import time

import datasets
from model import FOTSModel
from modules.parse_polys import parse_polys


def restore_checkpoint(folder, contunue,args=None):
    model = FOTSModel(args=args).to(torch.device("npu"))
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=0.001, weight_decay=1e-5)
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level = args.apex_opt_level, combine_grad=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=32, verbose=True, threshold=0.05, threshold_mode='rel')

    checkppoint_name = os.path.join(folder, 'epoch_8_checkpoint.pt')
    if os.path.isfile(checkppoint_name) and contunue:
        checkpoint = torch.load(checkppoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        # return 0, model, optimizer, lr_scheduler, +math.inf
        epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        best_score = checkpoint['best_score']
        return epoch, model, optimizer, lr_scheduler, best_score
    else:
        return 0, model, optimizer, lr_scheduler, +math.inf


def save_checkpoint(epoch, model, optimizer, lr_scheduler, best_score, folder, save_as_best):
    if not os.path.exists(folder):
        os.makedirs(folder)
    #if epoch > 60 and epoch % 6 == 0:
    if epoch==999:
    #if True:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_score': best_score  # not current score
        }, os.path.join(folder, 'epoch_{}_checkpoint.pt'.format(epoch)))

    if save_as_best:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_score': best_score  # not current score
        }, os.path.join(folder, 'best_checkpoint.pt'))
        print('Updated best_model')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_score': best_score  # not current score
    }, os.path.join(folder, 'last_checkpoint.pt'))


def fill_ohem_mask(raw_loss, ohem_mask, num_samples_total, max_hard_samples, max_rnd_samples):
    h, w = raw_loss.shape
    if num_samples_total != 0:
        top_val, top_idx = torch.topk(raw_loss.view(-1), num_samples_total)
        num_hard_samples = int(min(max_hard_samples, num_samples_total))

        num_rnd_samples = max_hard_samples + max_rnd_samples - num_hard_samples
        num_rnd_samples = min(num_rnd_samples, num_samples_total - num_hard_samples)
        weight = num_hard_samples + num_rnd_samples

        for id in range(min(len(top_idx), num_hard_samples)):
            val = top_idx[id]
            y = val // w
            x = val - y * w
            ohem_mask[y, x] = 1 #/ weight

        if num_rnd_samples != 0:
            for id in nprnd.randint(num_hard_samples, num_hard_samples + num_rnd_samples, num_rnd_samples):
                val = top_idx[id]
                y = val // w
                x = val - y * w
                ohem_mask[y, x] = 1 #/ weight


def detection_loss(pred, gt):
    y_pred_cls, y_pred_geo, theta_pred = pred
    y_true_cls, y_true_geo, theta_gt, training_mask = gt
    y_true_cls, theta_gt = y_true_cls.unsqueeze(1), theta_gt.unsqueeze(1)
    y_true_cls, y_true_geo, theta_gt = y_true_cls.to('npu'), y_true_geo.to('npu'), theta_gt.to('npu')

    raw_cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=y_pred_cls, target=y_true_cls, weight=None, reduction='none')

    d1_gt, d2_gt, d3_gt, d4_gt = torch.split(y_true_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred = torch.split(y_pred_geo, 1, 1)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_intersect = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
    h_intersect = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
    area_intersect = w_intersect * h_intersect
    area_union = area_gt + area_pred - area_intersect
    raw_tensor_loss = -torch.log((area_intersect+1) / (area_union+1)) + 10 * (1 - torch.cos(theta_pred - theta_gt))

    ohem_cls_mask = np.zeros(raw_cls_loss.shape, dtype=np.float32)
    ohem_reg_mask = np.zeros(raw_cls_loss.shape, dtype=np.float32)
    for batch_id in range(y_true_cls.shape[0]):
        y_true = y_true_cls[batch_id].squeeze().data.cpu().numpy().astype(np.uint8)
        mask = training_mask[batch_id].squeeze().data.cpu().numpy().astype(np.uint8)
        shrunk_mask = y_true & mask
        neg_mask = y_true.copy()
        neg_mask[y_true == 1] = 0
        neg_mask[y_true == 0] = 1
        neg_mask[mask == 0] = 0

        shrunk_sum = int(shrunk_mask.sum())
        if shrunk_sum != 0:
            ohem_cls_mask[batch_id, 0, shrunk_mask == 1] = 1 #/ shrunk_sum
        raw_loss = raw_cls_loss[batch_id].squeeze().data.cpu().numpy()
        raw_loss[neg_mask == 0] = 0
        raw_loss = torch.from_numpy(raw_loss)
        num_neg = int(neg_mask.sum())
        fill_ohem_mask(raw_loss, ohem_cls_mask[batch_id, 0], num_neg, 512, 512)

        raw_loss = raw_tensor_loss[batch_id].squeeze().data.cpu().numpy()
        raw_loss[shrunk_mask == 0] = 0
        raw_loss = torch.from_numpy(raw_loss)
        num_pos = int(shrunk_mask.sum())
        fill_ohem_mask(raw_loss, ohem_reg_mask[batch_id, 0], num_pos, 128, 128)

    if 0:
        for batch_id in range(y_true_cls.shape[0]):
            y_true = y_true_cls[batch_id].squeeze().data.cpu().numpy().astype(np.uint8)
            cv2.imshow('y_true', y_true*255)
            mask = training_mask[batch_id].squeeze().data.cpu().numpy().astype(np.uint8)
            cv2.imshow('mask', mask*255)

            shrunk_mask = y_true & mask
            cv2.imshow('shrunk pos', shrunk_mask*255)
            neg_mask = y_true.copy()
            neg_mask[y_true == 1] = 0
            neg_mask[y_true == 0] = 1
            neg_mask[mask == 0] = 0
            cv2.imshow('neg', neg_mask*255)

            cv2.imshow('ohem_cls', ohem_cls_mask[batch_id, 0])
            cv2.imshow('ohem_reg', ohem_reg_mask[batch_id, 0])

            cv2.waitKey()
    ohem_cls_mask_sum = int(ohem_cls_mask.sum())
    ohem_reg_mask_sum = int(ohem_reg_mask.sum())
    if 0 != ohem_cls_mask_sum:
        raw_cls_loss = raw_cls_loss * torch.from_numpy(ohem_cls_mask).npu()
        raw_cls_loss = raw_cls_loss.sum() / ohem_cls_mask_sum
    else:
        raw_cls_loss = 0

    if 0 != ohem_reg_mask_sum:
        raw_tensor_loss = raw_tensor_loss * torch.from_numpy(ohem_reg_mask).cuda()
        reg_loss = raw_tensor_loss.sum() / ohem_reg_mask_sum
    else:
        reg_loss = 0
    return reg_loss + raw_cls_loss


def show_tensors(cropped, classification, regression, thetas, training_mask, file_names):
    print(file_names[0])
    cropped = cropped[0].to('cpu').numpy()
    cropped = np.transpose(cropped, (1, 2, 0))
    cropped = cv2.resize(cropped, None, fx=0.25, fy=0.25) / 255

    d1, d2, d3, d4 = torch.split(regression.to('cpu'), 1, 1)
    d1, d2, d3, d4 = d1[0].view(160, 160).detach().numpy(), d2[0].view(160, 160).detach().numpy(), d3[0].view(160, 160).detach().numpy(), d4[0].view(160, 160).detach().numpy()

    thetas = thetas[0].view(160, 160).to('cpu').detach().numpy()

    cv2.imshow('', cropped)
    cv2.waitKey(0)
    cv2.imshow('', classification[0].view(160, 160).to('cpu').detach().numpy())
    cv2.waitKey(0)
    cv2.imshow('', d1 / np.amax(d1))
    cv2.waitKey(0)
    cv2.imshow('', d2 / np.amax(d2))
    cv2.waitKey(0)
    cv2.imshow('', d3 / np.amax(d3))
    cv2.waitKey(0)
    cv2.imshow('', d4 / np.amax(d4))
    cv2.waitKey(0)
    cv2.imshow('', thetas / np.amin(thetas))
    cv2.waitKey(0)
    cv2.imshow('', training_mask[0].to('cpu').detach().numpy())
    cv2.waitKey(0)


def fit(start_epoch, model, loss_func, opt, lr_scheduler, best_score, max_batches_per_iter_cnt, checkpoint_dir, train_dl, valid_dl):
    batch_per_iter_cnt = 0
    #epochtime=time.time()
    #for epoch in range(start_epoch, 999):
    for epoch in range(start_epoch,args.max_epochs):
        model.train()
        train_loss_stats = 0.0
        loss_count_stats = 0
        pbar = tqdm.tqdm(train_dl, 'Epoch ' + str(epoch), ncols=80)
        iter = 0
        #for i, sample in enumerate(train_dl):
              #cropped, classification, regression, thetas, training_mask = sample
        for cropped, classification, regression, thetas, training_mask in pbar:
            # if iter == 12: exit()
            # print("iter===================", iter)
            start_time=time.time()
            if batch_per_iter_cnt == 0:
                optimizer.zero_grad()
            prediction = model(cropped.to('npu'))

            if 0:
                for batch_id in range(cropped.shape[0]):
                    img = cropped[batch_id].data.cpu().numpy().transpose((1, 2, 0)) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                    cv2.imshow('img', img[:, :, ::-1])

                    cls = np.squeeze(prediction[0][batch_id].data.cpu().numpy())
                    #cls = cv2.resize(cls, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_AREA)

                    mask = training_mask[batch_id].data.cpu().numpy()
                    #mask = cv2.resize(mask, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_AREA)
                    cv2.imshow('mask', mask)
                    #cv2.imshow('cls', cls*mask)
                    cls_bin = cls > 0.5

                    cls2 = cls.copy()
                    cls2[cls_bin != True] = 0

                    cv2.imshow('cls', cls)

                    #res = parse_polys(cls2,
                    #                  prediction[1][batch_id].data.cpu().numpy(),
                    #                  np.squeeze(prediction[2][batch_id].data.cpu().numpy()), img=img.copy())

                    #top_dist = regression[batch_id, 0].data.cpu().numpy()
                    #top_dist /= top_dist.max()
                    ##top_dist = cv2.resize(top_dist, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_AREA)
                    #cv2.imshow('top_dist', top_dist)
                    #
                    #right_dist = regression[batch_id, 1].data.cpu().numpy()
                    #right_dist /= right_dist.max()
                    ##right_dist = cv2.resize(right_dist, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_AREA)
                    #cv2.imshow('right_dist', right_dist)
                    #
                    #bottom_dist = regression[batch_id, 2].data.cpu().numpy()
                    #bottom_dist /= bottom_dist.max()
                    ##bottom_dist = cv2.resize(bottom_dist, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_AREA)
                    #cv2.imshow('bottom_dist', bottom_dist)
                    #
                    #left_dist = regression[batch_id, 3].data.cpu().numpy()
                    #left_dist /= left_dist.max()
                    ##left_dist = cv2.resize(left_dist, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_AREA)
                    #cv2.imshow('left_dist', left_dist)
                    #
                    #
                    ##angle = thetas[batch_id].data.cpu().numpy()
                    #angle = prediction[2][batch_id].squeeze().data.cpu().numpy()
                    ##angle /= angle.max()
                    ##left_dist = cv2.resize(left_dist, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_AREA)
                    #cv2.imshow('angle', (angle * cls_bin / np.pi * 180).astype(np.uint8))
                    cv2.waitKey()

            # show_tensors(cropped, classification, regression, thetas, training_mask, file_names)

            # show_tensors(cropped, *prediction, training_mask, file_names)

            loss = loss_func(prediction, (classification, regression, thetas, training_mask)) / max_batches_per_iter_cnt
            train_loss_stats += loss.item()
            if args.apex:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            batch_per_iter_cnt += 1
            if batch_per_iter_cnt == max_batches_per_iter_cnt:
                opt.step()
                batch_per_iter_cnt = 0
                loss_count_stats += 1
                mean_loss = train_loss_stats / loss_count_stats
                #pbar.set_postfix({'Mean loss': f'{mean_loss:.5f}'}, refresh=False)
                end_time=time.time()
                print('Epoch {} Mean loss {:.10f}, step_time {:.3f}\n'.format(epoch, mean_loss, end_time - start_time))
            iter = iter + 1
        lr_scheduler.step(mean_loss, epoch)

        if valid_dl is None:
            val_loss = train_loss_stats / loss_count_stats
        else:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_loss_count = 0
                for cropped, classification, regression, thetas, training_mask, file_names in valid_dl:
                    prediction = model(cropped.to('npu'))
                    loss = loss_func(prediction, (classification, regression, thetas, training_mask, file_names))
                    val_loss += loss.item()
                    val_loss_count += len(cropped)
            val_loss /= val_loss_count
        # print('Val loss: ', val_loss)

        if best_score > val_loss:
            best_score = val_loss
            save_as_best = True
        else:
            save_as_best = False
        #save_checkpoint(epoch, model, opt, lr_scheduler, best_score, checkpoint_dir, save_as_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apex', action='store_true',help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level',default='O1',type=str,help='For apex mixed precision training, O0 for FP32 training, O1 for mixed precision training')
    parser.add_argument('--train-folder', type=str, required=True, help='Path to folder with train images and labels')
    parser.add_argument('--batch-size', type=int, default=21, help='Number of batches to process before train step')
    parser.add_argument('--batches-before-train', type=int, default=2, help='Number of batches to process before train step')
    parser.add_argument('--num-workers', type=int, default=8, help='Path to folder with train images and labels')
    parser.add_argument('--continue-training', action='store_true', help='continue training')
    parser.add_argument('--max-epochs', type=int, default=2, help='max epochs')
    args = parser.parse_args()

    #data_set = datasets.SynthText(args.train_folder, datasets.transform)
    data_set = datasets.ICDAR2015(args.train_folder, datasets.transform)

    # SynthText and ICDAR2015 have different layouts. One will probably need to provide two different paths to train
    # on concatination of these two data sets. But the paper doesn't concat them so me neither
    # datai_set = torch.utils.data.ConcatDataset((synth, icdar))

    dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                     sampler=None, batch_sampler=None, num_workers=args.num_workers)
    checkoint_dir = 'runs'
    epoch, model, optimizer, lr_scheduler, best_score = restore_checkpoint(checkoint_dir, args.continue_training,args=args)
    model = torch.nn.DataParallel(model)
    fit(epoch, model, detection_loss, optimizer, lr_scheduler, best_score, args.batches_before_train, checkoint_dir, dl, None)
