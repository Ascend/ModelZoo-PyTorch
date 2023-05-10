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

import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable
from mmseg.evaluation import IoUMetric


@staticmethod
def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                        num_classes: int, ignore_index: int):
    """Calculate Intersection and Union.

    Args:
        pred_label (torch.tensor): Prediction segmentation map
            or predict result filename. The shape is (H, W).
        label (torch.tensor): Ground truth segmentation map
            or label filename. The shape is (H, W).
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

    Returns:
        torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
        torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
        torch.Tensor: The prediction histogram on all classes.
        torch.Tensor: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]

    # torch.histc is currently not supported on NPU
    # and needs to be calculated on the CPU
    area_intersect = torch.histc(
        intersect.float().cpu(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_pred_label = torch.histc(
        pred_label.float().cpu(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_label = torch.histc(
        label.float().cpu(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


IoUMetric.intersect_and_union = intersect_and_union
