#
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
#
import torch
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import numpy as np
from config import coordinates_cat, proposalN, set, vis_num
from util.cal_iou import calculate_iou
from util.vis import image_with_boxes
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
	NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
	torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def eval(model, testloader, criterion, status, save_path, epoch):
    model.eval()
    print('Evaluating')

    raw_loss_sum = 0
    local_loss_sum = 0
    windowscls_loss_sum = 0
    total_loss_sum = 0
    iou_corrects = 0
    raw_correct = 0
    local_correct = 0

    with torch.no_grad():
        n = 0
        for i, data in enumerate(tqdm(testloader)):
            n = n + 1
            if n >= 5:
                pass
            if set == 'CUB':
                images, labels, boxes, scale = data
            else:
                images, labels = data
            images = images.npu()
            labels = labels.npu()

            proposalN_windows_score,proposalN_windows_logits, indices, \
            window_scores, coordinates, raw_logits, local_logits, local_imgs = model(images, epoch, i, status)

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                                        labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            total_loss = raw_loss + local_loss + windowscls_loss

            raw_loss_sum += raw_loss.item()
            local_loss_sum += local_loss.item()
            windowscls_loss_sum += windowscls_loss.item()

            total_loss_sum += total_loss.item()

            if set == 'CUB':
                # computer resized coordinates of boxes
                boxes_coor = boxes.float()
                resized_boxes = torch.cat([(boxes_coor[:,0] * scale[:, 0]).unsqueeze(1) ,(boxes_coor[:,1] * scale[:, 1]).unsqueeze(1),
                                           (boxes_coor[:,2] * scale[:, 0]).unsqueeze(1), (boxes_coor[:,3] * scale[:, 1]).unsqueeze(1)], dim=1)
                resized_coor = torch.cat([resized_boxes[:,0].unsqueeze(1) ,resized_boxes[:,1].unsqueeze(1),
                                           (resized_boxes[:,0] + resized_boxes[:,2]).unsqueeze(1), (resized_boxes[:,1]+resized_boxes[:,3]).unsqueeze(1)], dim=1).round().int()


                iou = calculate_iou(coordinates.cpu().numpy(), resized_coor.numpy())
                iou_corrects += np.sum(iou >= 0.5)

            # correct num
            # raw
            pred = raw_logits.max(1, keepdim=True)[1]
            raw_correct += pred.eq(labels.view_as(pred)).sum().item()
            # local
            pred = local_logits.max(1, keepdim=True)[1]
            local_correct += pred.eq(labels.view_as(pred)).sum().item()

            # raw branch tensorboard
            if i == 0:
                if set == 'CUB':
                    box_coor = resized_coor[:vis_num].numpy()
                    pred_coor = coordinates[:vis_num].cpu().numpy()
                    with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment=status + 'raw') as writer:
                        cat_imgs = []
                        for j, coor in enumerate(box_coor):
                            img = image_with_boxes(images[j], [coor])
                            img = image_with_boxes(img, [pred_coor[j]], color=(0, 255, 0))
                            cat_imgs.append(img)
                        cat_imgs = np.concatenate(cat_imgs, axis=1)
                        writer.add_images(status + '/' + 'raw image with boxes', cat_imgs, epoch, dataformats='HWC')

            # object branch tensorboard
            if i == 0:
                indices_ndarray = indices[:vis_num,:proposalN].cpu().numpy()
                with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment=status + 'object') as writer:
                    cat_imgs = []
                    for j, indice_ndarray in enumerate(indices_ndarray):
                        img = image_with_boxes(local_imgs[j], coordinates_cat[indice_ndarray])
                        cat_imgs.append(img)
                    cat_imgs = np.concatenate(cat_imgs, axis=1)
                    writer.add_images(status + '/' + 'object image with windows', cat_imgs, epoch, dataformats='HWC')

            # if status == 'train':
            #     if i >= 2 :
            #         break

    raw_loss_avg = raw_loss_sum / (i+1)
    local_loss_avg = local_loss_sum / (i+1)
    windowscls_loss_avg = windowscls_loss_sum / (i+1)
    total_loss_avg = total_loss_sum / (i+1)

    raw_accuracy = raw_correct / len(testloader.dataset)
    local_accuracy = local_correct / len(testloader.dataset)
    print('len(testloader.dataset): %d' % (len(testloader.dataset)))

    return raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
           local_loss_avg