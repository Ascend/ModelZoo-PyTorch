# Copyright 2021 Huawei Technologies Co., Ltd
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

import os, sys
import numpy as np
import torch
import pathlib
sys.path.append(r"./pytorch-ssd")
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.data_preprocessing import PredictionTransform
import vision.utils.box_utils as box_utils
import vision.utils.measurements as measurements


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurem

if __name__ == "__main__":
    dataroot=os.path.abspath(sys.argv[1])
    label_file = os.path.abspath(sys.argv[2])
    class_names = [name.strip() for name in open(label_file).readlines()]
    npu_result = os.path.abspath(sys.argv[3])
    eval_path = os.path.abspath(sys.argv[4])
    eval_path = pathlib.Path(eval_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    dataset = VOCDataset(dataroot, is_test=True)
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    size = 300
    mean = np.array([127, 127, 127])  # RGB layout
    std = 128.0
    iou_threshold = 0.45
    prob_threshold=0.01
    candidate_size=200
    sigma=0.5

    results = []
    for i in range(len(dataset)):
        print("i:",i)
        image = dataset.get_image(i)
        image_id = dataset.ids[i]
        height, width, _ = image.shape
        scores_id = str(image_id)+'_0.bin'
        boxes_id = str(image_id)+'_1.bin'

        boxes = np.fromfile(os.path.join(npu_result, boxes_id), dtype='float32').reshape((1,3000,4))
        scores = np.fromfile(os.path.join(npu_result, scores_id), dtype='float32').reshape((1,3000,21))        
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        boxes = boxes[0]
        scores = scores[0]

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs_ = box_utils.nms(box_probs, "hard",
                                      score_threshold=prob_threshold,
                                      iou_threshold=iou_threshold,
                                      sigma=sigma,
                                      top_k=-1,
                                      candidate_size=candidate_size)
            picked_box_probs.append(box_probs_)
            picked_labels.extend([class_index] * box_probs_.size(0))
        if not picked_box_probs:
            print("###########################################")
            boxes_, labels_, probs_ =  torch.tensor([]), torch.tensor([]), torch.tensor([])
        else:
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, 0] *= width
            picked_box_probs[:, 1] *= height
            picked_box_probs[:, 2] *= width
            picked_box_probs[:, 3] *= height
            boxes_, labels_, probs_ = picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
        indexes = torch.ones(labels_.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels_.reshape(-1, 1).float(),
            probs_.reshape(-1, 1),
            boxes_ + 1.0  # matlab's indexes start from 1
            ],dim=1 ))
        #print(results)
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    average_path = eval_path / "average.txt"
    fa = open(average_path, "w")
    fa.write("Average Precision Per-class:\n\n")
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            iou_threshold,
            use_2007_metric = True
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")
        fa.write(f"{class_name}: {ap}\n")
    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    fa.write(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    fa.close()