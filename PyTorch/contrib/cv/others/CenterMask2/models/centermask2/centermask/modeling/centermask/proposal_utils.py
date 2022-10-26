# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI) in 28/01/2020.
import math
import torch

from detectron2.structures import Instances
from detectron2.structures import Boxes

def add_ground_truth_to_proposals(targets, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        targets(list[Instances]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert targets is not None

    assert len(proposals) == len(targets)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(tagets_i, proposals_i)
        for tagets_i, proposals_i in zip(targets, proposals)
    ]


def add_ground_truth_to_proposals_single_image(targets_i, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with targets and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.scores.device
    proposals.proposal_boxes = proposals.pred_boxes
    proposals.remove("pred_boxes")
    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(targets_i), device=device)
    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = targets_i.gt_boxes
    # to have the same fields with proposals
    gt_proposal.scores = gt_logits
    gt_proposal.pred_classes = targets_i.gt_classes
    gt_proposal.locations = torch.ones((len(targets_i), 2), device=device)
    
    '''try to fix dynamic shape problem'''
    proposals_new=Instances(proposals.image_size)

    fix_num=108
    proposals_num = len(proposals)
    #import pdb; pdb.set_trace()
    proposals_new.proposal_boxes = (torch.zeros((fix_num,4), device=device)).float()
    proposals_boxes_indx = torch.tensor([[i,i,i,i] for i in range(0,proposals_num)],device=device)
    proposals_new.proposal_boxes = Boxes(proposals_new.proposal_boxes.scatter(0,proposals_boxes_indx,proposals.proposal_boxes.tensor))
    proposals_new.scores = (gt_logit_value * torch.ones(fix_num, dtype = proposals.scores.dtype, device = device))
    proposals_scores_indx = torch.tensor(list(range(0,proposals_num)),device=device)
    proposals_new.scores = proposals_new.scores.scatter(0,proposals_scores_indx,proposals.scores)
    proposals_new.pred_classes = (80 * torch.ones(fix_num, dtype = torch.int32, device = device))
    proposals_new.pred_classes = proposals_new.pred_classes.scatter(0, proposals_scores_indx, proposals.pred_classes)
    proposals_new.locations = torch.ones((fix_num, 2), device=device, dtype= torch.float)
    proposals_locations_indx = torch.tensor([[i, i] for i in range(0, proposals_num)],device=device)
    proposals_new.locations = proposals_new.locations.scatter(0,proposals_locations_indx,proposals.locations)
    new_proposals = Instances.cat([proposals_new, gt_proposal])

    return new_proposals
