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
# limitations under the License.ls
import sys

sys.path.append(r'./detr')
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.coco_eval import CocoEvaluator
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from hubconf import detr_resnet50
from models.detr import PostProcess
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/opt/npu/coco/')
    parser.add_argument('--result', default='result',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    return parser


def main(args):
    device = torch.device(args.device)
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=2)
    base_ds = get_coco_api_from_dataset(dataset_val)
    postprocessors = {'bbox': PostProcess()}
    print('start validate')

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    print_freq = 10
    path = args.result
    files = os.listdir('{}/val2017'.format(args.coco_path))
    files.sort()
    for file, (samples, targets) in zip(files, metric_logger.log_every(data_loader_val, print_freq, header)):
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        pred_boxes_file = os.path.join(path, file.replace('.jpg', '_1.bin'))
        pred_boxes = np.fromfile(pred_boxes_file, dtype=np.float32)
        pred_boxes.shape = 1, 100, 4
        pred_logits_file = os.path.join(path, file.replace('.jpg', '_0.bin'))
        pred_logits = np.fromfile(pred_logits_file, dtype=np.float32)
        pred_logits.shape = 1, 100, 92
        om_out = {'pred_logits': torch.from_numpy(pred_logits),
                  'pred_boxes': torch.from_numpy(pred_boxes)}
        outputs = om_out
        metric_logger.update(loss=0.7)
        metric_logger.update(class_error=0.5)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)