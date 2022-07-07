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
"""Validation function"""

import time
import logging
import numpy as np

import torch
from torch import nn

from helpers.miou_utils import compute_iu, compute_ius_accs, fast_cm
from helpers.utils import try_except
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


logger = logging.getLogger(__name__)


@try_except
def validate(
    segmenter,
    val_loader,
    epoch,
    epoch2,
    num_classes=-1,
    print_every=10,
    omit_classes=[0],
):
    """Validate segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      val_loader (DataLoader) : training data iterator
      epoch (int) : current search epoch
      epoch2 (int) : current segm. training epoch
      num_classes (int) : number of segmentation classes
      print_every (int) : how often to print out information
      omit_classes (list of int) : indices of classes to ignore when computing metrics

    Returns:
      Reward (float)

    """
    try:
        val_loader.dataset.set_stage("val")
    except AttributeError:
        val_loader.dataset.dataset.set_stage("val")  # for subset
    segmenter.eval()

    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        n=0
        for i, sample in enumerate(val_loader):
            if n==1:
                pass
            n=n+1
            image = sample["image"]
            target = sample["mask"]
            input_var = torch.autograd.Variable(image).float().npu()
            # Compute output
            output = segmenter(input_var)
            if isinstance(output, tuple):
                output, _ = output
            output = nn.Upsample(
                size=target.size()[1:], mode="bilinear", align_corners=False
            )(output)
            # Compute IoU
            output = output.data.cpu().numpy().argmax(axis=1).astype(np.uint8)
            gt = target.data.cpu().numpy().astype(np.uint8)
            # Ignore every class index larger than the number of classes
            gt_idx = gt < num_classes
            cm += fast_cm(output[gt_idx], gt[gt_idx], num_classes)

            if i % print_every == 0:
                logger.info(
                    " Val epoch: {} [{}/{}]\t"
                    "Mean IoU: {:.3f}".format(
                        epoch,
                        i,
                        len(val_loader),
                        np.mean([iu for iu in compute_iu(cm) if iu <= 1.0]),
                    )
                )
    ious, n_pixels, accs = compute_ius_accs(cm)
    #logger.info(" IoUs: {}, accs: {}".format(ious, accs))
    # IoU by default is 2, so we ignore all the unchanged classes
    present_ind = np.array([idx for idx, iu in enumerate(ious) if iu <= 1.0])
    # And ignore classes that might skew the evaluation metrics (e.g., background)
    present_ind = np.setdiff1d(present_ind, omit_classes)
    present_ious = ious[present_ind]
    present_pixels = n_pixels[present_ind]
    present_accs = accs[present_ind]
    miou = np.mean(present_ious)
    macc = np.mean(present_accs)
    mfwiou = np.sum(present_ious * present_pixels) / np.sum(present_pixels)
    metrics = [miou, macc, mfwiou]
    reward = np.prod(metrics) ** (1.0 / len(metrics))
    info = (
        " Val epoch: {}/{}\tMean IoU: {:.3f}\tMean FW-IoU: {:.3f}\t"
        "Mean Acc: {:.3f}\tReward: {:.3f}"
    ).format(epoch, epoch2, miou, mfwiou, macc, reward)
    logger.info(info)
    return reward
