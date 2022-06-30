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
import numpy as np


def object_detection_collate(batch):
    images = []
    gt_boxes = []
    gt_labels = []
    image_type = type(batch[0][0])
    box_type = type(batch[0][1])
    label_type = type(batch[0][2])
    for image, boxes, labels in batch:
        if image_type is np.ndarray:
            images.append(torch.from_numpy(image))
        elif image_type is torch.Tensor:
            images.append(image)
        else:
            raise TypeError(f"Image should be tensor or np.ndarray, but got {image_type}.")
        if box_type is np.ndarray:
            gt_boxes.append(torch.from_numpy(boxes))
        elif box_type is torch.Tensor:
            gt_boxes.append(boxes)
        else:
            raise TypeError(f"Boxes should be tensor or np.ndarray, but got {box_type}.")
        if label_type is np.ndarray:
            gt_labels.append(torch.from_numpy(labels))
        elif label_type is torch.Tensor:
            gt_labels.append(labels)
        else:
            raise TypeError(f"Labels should be tensor or np.ndarray, but got {label_type}.")
    return torch.stack(images), gt_boxes, gt_labels