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
import time
import torch
from torch import nn
import torch.nn.functional as F


class MultiTaskLossParallel(nn.Module):
    def __init__(self, opt, config, heatmap_weight=1, offset_weight=1, **kwargs):
        super(MultiTaskLossParallel, self).__init__()
        self.nstack = opt.nstack
        self.batch_size = opt.batch_size
        self.offset_start = config.offset_start
        self.multi_task_weight = opt.multi_task_weight
        self.scale_weight = opt.scale_weight
        self.nstack_weight = opt.nstack_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight

    def forward(self, pred_tuple, target_tuple):
        """
        Compute the multi-task total loss
        :param pred_tuple: [nstack * [(bacth,C,128,128), (bacth,C,64,64), (bacth,C,32,32),
        (bacth,C,16,16)], (bacth,C,8,8)]
        :param target_tuple: target tensors, i.e.,
         mask_misses,   heatmaps,       offsets,       mask_offsets,
        [batch,1,128,128], [batch,44,128,128], [batch,36,128,128], [batch,36,128,128]
        :return: scalar tensor
        """
        # we use 4 stacks, 5 scales
        # TODO: 鏄敤5涓笉鍚宻cale濂借繕鏄4涓猻cale鐩戠潱濂斤紵
        pred_scale_tensors = [torch.cat([pred_tuple[j][i][None, ...] for j in range(self.nstack)], dim=0) for i in
                              range(5)]  # concatenate the same scale output of different stacks
        # different scale losses have different order of magnitudes owning to different pixel numbers (feature map size)
        loss_scales = [self._loss_per_scale(pred_scale_tensors[i], target_tuple) * self.scale_weight[i] for i in
                       range(5)]  # different scale losses of all nstack
        loss_per_batch = sum(loss_scales) / sum(self.scale_weight)
        return loss_per_batch  # should divide the batch size in the main train

    def _loss_per_scale(self, pred, target):
        """
        Compute the loss on a particular scale.
        :param pred: tensor (nstack, bacth, C, H, W)
        :param target: mask_misses, heatmaps, offsets, mask_offsets of shape (N, C, H, W)
        :return:
        """
        # TODO锛 娌℃湁骞宠　keypoint 鍜 body part涓ら儴鍒嗘崯澶憋紝鍙互鍦ㄨ繖閲屾妸heatmap杩涗竴姝ユ媶鍒
        pred_heatmap = pred[:, :, :self.offset_start]
        # pred_offset = pred[:, :, self.offset_start:]

        gt_mask_misses = F.interpolate(target[0], size=pred.shape[-2:], mode='bilinear')
        gt_heatmaps = F.adaptive_avg_pool2d(target[1], output_size=pred.shape[-2:])

        # gt_offsets = F.adaptive_avg_pool2d(target[2], output_size=pred.shape[-2:])
        # gt_mask_offsets = F.interpolate(target[3], size=pred.shape[-2:], mode='bilinear')

        #
        #   F.adaptive_max_pool2d(target[3], output_size=pred.shape[-2:])

        # ############# For debug ##############################
        # heatmap = gt_heatmaps[0,...].cpu().numpy().squeeze()
        # offset = gt_mask_offsets[0,...].cpu().numpy().squeeze()
        #
        # import matplotlib.pylab as plt
        # # plt.imshow(heatmap[11,:,:]) # mask_all
        # plt.imshow(heatmap[43, :,:])  # mask_all
        # plt.show()
        # #####################################################
        heatmap_loss = self.l2_loss(pred_heatmap, gt_heatmaps[None, ...], gt_mask_misses[None, ...]
                                          , nstack_weight=self.nstack_weight)
        # offset_loss = self.l1_loss(pred_offset, gt_offsets[None, ...], gt_mask_offsets[None, ...],
        #                            nstack_weight=self.nstack_weight)
        #
        # multi_task_loss = heatmap_loss * self.multi_task_weight[0] + offset_loss * self.multi_task_weight[1]
        # return multi_task_loss / sum(self.multi_task_weight)
        return heatmap_loss

    @staticmethod
    def focal_l2_loss(s, sxing, mask_miss, gamma=2, nstack_weight=[1, 1, 1, 1]):
        """
        Compute the focal L2 loss between predicted and groundtruth score maps.
        :param s:  predicted tensor (nstack, batch, channel, height, width), predicted score maps
        :param sxing: target tensor (nstack, batch, channel, height, width)
        :param mask_miss: tensor (1, batch, 1, height, width)
        :param gamma: focusing parameter
        :return: a scalar tensor
        """
        # eps = 1e-8  # 1e-12
        # s = torch.clamp(s, eps, 1. - eps)  # improve the stability of the focal loss
        st = torch.where(torch.ge(sxing, 0.01), s, 1 - s)
        factor = (1. - st) ** gamma
        # multiplied by mask_miss via broadcast operation
        out = (s - sxing) ** 2 * factor * mask_miss  # type: torch.Tensor
        # sum over the feature map, should divide by batch_size afterwards
        loss_nstack = out.sum(dim=(1, 2, 3, 4))   # losses from nstack 1, 2, 3, 4...
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        print(' heatmap focal L2 loss per stack..........  ', loss_nstack.detach().cpu().numpy())
        weight_loss = [loss_nstack[i] * nstack_weight[i] for i in range(len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)
        return loss

    @staticmethod
    def l1_loss(pred, target, mask_offset, nstack_weight=[1, 1, 1, 1]):
        """
        Compute the  L1 loss of offset feature maps
        :param pred: predicted tensor (nstack, batch, channel, height, width), predicted feature maps
        :param target: target tensor (nstack, batch, channel, height, width)
        :param mask_offset: tensor (nstack, batch, channel, height, width)
        :param nstack_weight:
        :return:
        """
        out = torch.abs(pred - target) * mask_offset  # type: torch.Tensor
        # sum over the feature map, should divide by batch afterwards
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        print(' offset L1 loss per stack >>>>>>>>  ', loss_nstack.detach().cpu().numpy())
        weight_loss = [loss_nstack[i] * nstack_weight[i] for i in range(len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)  # normalized loss by weights
        return loss

    @staticmethod
    def l2_loss(s, sxing, mask_miss, nstack_weight=[1, 1, 1, 1]):
        """
        Compute the L2 loss between predicted and groundtruth score maps.
        :param s:  predicted tensor (nstack, batch, channel, height, width), predicted score maps
        :param sxing: target tensor (nstack, batch, channel, height, width)
        :param mask_miss: tensor (nstack, batch, 1, height, width)
        :return: a scalar tensor
        """
        # multiplied by mask_miss via broadcast operation
        # eps = 1e-8  # 1e-12  #
        # s = torch.clamp(s, eps, 1 - eps)
        out = (s - sxing) ** 2 * mask_miss  # type: torch.Tensor
        # sum over the feature map, should divide by batch afterwards
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        print(' heatmap L2 loss per stack.........  ', loss_nstack.detach().cpu().numpy())
        weight_loss = [loss_nstack[i] * nstack_weight[i] for i in range(len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)
        return loss


