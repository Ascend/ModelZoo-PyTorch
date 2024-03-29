# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
from collections import Counter


def gt_label2entity(gt_infos):
    """Get all entities from ground truth infos.
    Args:
        gt_infos (list[dict]): Ground-truth information contains text and
            label.
    Returns:
        gt_entities (list[list]): Original labeled entities in groundtruth.
                    [[category,start_position,end_position]]
    """
    gt_entities = []
    for gt_info in gt_infos:
        line_entities = []
        label = gt_info['label']
        for key, value in label.items():
            for _, places in value.items():
                for place in places:
                    line_entities.append([key, place[0], place[1]])
        gt_entities.append(line_entities)
    return gt_entities


def _compute_f1(origin, found, right):
    """Calculate recall, precision, f1-score.

    Args:
        origin (int): Original entities in groundtruth.
        found (int): Predicted entities from model.
        right (int): Predicted entities that
                        can match to the original annotation.
    Returns:
        recall (float): Metric of recall.
        precision (float): Metric of precision.
        f1 (float): Metric of f1-score.
    """
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (
        precision + recall)
    return recall, precision, f1


def compute_f1_all(pred_entities, gt_entities):
    """Calculate precision, recall and F1-score for all categories.

    Args:
        pred_entities: The predicted entities from model.
        gt_entities: The entities of ground truth file.
    Returns:
        class_info (dict): precision,recall, f1-score in total
                        and each categories.
    """
    origins = []
    founds = []
    rights = []
    for i, _ in enumerate(pred_entities):
        origins.extend(gt_entities[i])
        founds.extend(pred_entities[i])
        rights.extend([
            pre_entity for pre_entity in pred_entities[i]
            if pre_entity in gt_entities[i]
        ])

    class_info = {}
    origin_counter = Counter([x[0] for x in origins])
    found_counter = Counter([x[0] for x in founds])
    right_counter = Counter([x[0] for x in rights])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = _compute_f1(origin, found, right)
        class_info[type_] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    recall, precision, f1 = _compute_f1(origin, found, right)
    class_info['all'] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }
    return class_info


def eval_ner_f1(results, gt_infos):
    """Evaluate for ner task.

    Args:
        results (list): Predict results of entities.
        gt_infos (list[dict]): Ground-truth information which contains
                            text and label.
    Returns:
        class_info (dict): precision,recall, f1-score of total
                            and each catogory.
    """
    assert len(results) == len(gt_infos)
    gt_entities = gt_label2entity(gt_infos)
    pred_entities = []
    for i, gt_info in enumerate(gt_infos):
        line_entities = []
        for result in results[i]:
            line_entities.append(result)
        pred_entities.append(line_entities)
    assert len(pred_entities) == len(gt_entities)
    class_info = compute_f1_all(pred_entities, gt_entities)

    return class_info
