# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import yaml
import onnx
from onnx import helper
import torch
import torch.nn as nn
import torchvision
import numpy as np


class YOLOPOSTPROCESS(nn.Module):
    def __init__(
            self,
            score_thresh,
            nms_thresh,
            detection_per_img
    ):
        super().__init__()

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img
        self.num_classes = 80

    def forward(self, prediction):
        grids = []
        expanded_strides = []
        img_size = [640, 640]
        strides = [8, 16, 32]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = torch.meshgrid(torch.arange(wsize), torch.arange(hsize))
            grid = torch.stack((xv, yv), 2).reshape(1, -1, 2)

            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1)

        real_grids = torch.ones_like(grids)
        real_grids[0][:, 0] = grids[0][:, 1]
        real_grids[0][:, 1] = grids[0][:, 0]

        expanded_strides = torch.cat(expanded_strides, dim=1)
        center_xy = (prediction[:, :, :2] + real_grids) * expanded_strides
        width_height = torch.exp(prediction[:, :, 2:4]) * expanded_strides

        # 该步操作主要是将bbox格式从[x_center, y_center, width, height]变为onnx的nms算子要求的[y1 ,x1, y2 ,x2]
        # center_x, center_y, w, h -> upper_left_y, upper_left_x, right_y, right_x
        y1 = center_xy[:, :, 1] - width_height[:, :, 1] / 2.  # y1
        x1 = center_xy[:, :, 0] - width_height[:, :, 0] / 2.  # x1
        y2 = center_xy[:, :, 1] + width_height[:, :, 1] / 2.  # y2
        x2 = center_xy[:, :, 0] + width_height[:, :, 0] / 2.  # x2
        bbox_review = torch.cat([y1.unsqueeze(2), x1.unsqueeze(2), y2.unsqueeze(2), x2.unsqueeze(2)], dim=2)

        # tensor.to(tensor)并不是广播成与括号中tensor的shape一致，而是to(device)的操作，意为将俩者同放到device或者host侧
        # Get score and class with highest confidence
        # 取出后80个数据中的最大值数据和该数据的索引
        class_conf, class_pred = torch.max(prediction[:, :, 5: 5 + self.num_classes], 2, keepdim=True)
        scores = torch.mul(prediction[:, :, 4:5], class_conf)
        # export expand operator to onnx more nicely

        # 如果你不想每个类别都做一次nms,而是所有类别一起做nms
        # 就需要把不同类别的目标框尽量没有重合，不至于把不同类别的IOU大的目标框滤掉
        # 先用每个类别id乘一个很大的数，可以取bbox坐标的最大值作为offset，但是由于会引入算子，故取常量，把每个类别的bbox坐标都加上相应的offset，
        # 从而实现batched nms
        offsets = class_pred.to(bbox_review) * torch.tensor(700, dtype=torch.int32)
        bbox_add_offsets = bbox_review + offsets

        return bbox_review, scores, class_pred.float(), bbox_add_offsets
