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

from yolact_edge.data import COCODetection
from yolact_edge.utils.augmentations import BaseTransform
import numpy as np
import json
import os
import torch
from tqdm import tqdm
import argparse
from yolact_edge.yolact import Yolact
from yolact_edge.data import set_cfg
from yolact_edge.data import cfg
from yolact_edge.layers.output_utils import postprocess
from yolact_edge.layers.box_utils import jaccard
from eval import APDataObject
from collections import OrderedDict

iou_thresholds = [x / 100 for x in range(50, 100, 5)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--top_k', default=5)
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--fast_nms', default=True)
    parser.add_argument('--eval_stride', default=5)
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true')
    parser.add_argument('--web_det_path', default='web/dets/')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true')
    parser.add_argument('--display_lincomb', default=False)
    parser.add_argument('--benchmark', default=False)
    parser.add_argument('--fast_eval', default=False)
    parser.add_argument('--deterministic', default=False)
    parser.add_argument('--no_sort', default=False)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--mask_proto_debug', default=False)
    parser.add_argument('--no_crop', default=False)
    parser.add_argument('--image', default=None)
    parser.add_argument('--images', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--video_multiframe', default=1)
    parser.add_argument('--score_threshold', default=0)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--yolact_transfer', dest='yolact_transfer', action='store_true')
    parser.add_argument('--coco_transfer', dest='coco_transfer', action='store_true')
    parser.add_argument('--drop_weights', default=None)
    parser.add_argument('--calib_images', default=None)
    parser.add_argument('--trt_batch_size', default=1)
    parser.add_argument('--disable_tensorrt', default=False)
    parser.add_argument('--use_fp16_tensorrt', default=False)
    parser.add_argument('--use_tensorrt_safe_mode', default=False)

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

    parser.add_argument('--trained_model',
                        default='yolact_edge_resnet50_54_800000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--config', default='yolact_edge_resnet50_config',
                        help='The config object to use.')
    parser.add_argument('--image_path', default='./data/coco/val2017', help='The folder path of coco images.')
    parser.add_argument('--json_file', default='./data/coco/annotations/instances_val2017.json', help='The path of coco json file.')
    parser.add_argument('--file_path', required=True, help='The floder path of output files.')
    parser.add_argument('--ids_path', default='./inputs/ids.json', help='The path of json file which stroes dataset.ids.')

    global args
    args = parser.parse_args(argv)


def mask_iou(mask1, mask2, iscrowd=False):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """

    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    if iscrowd:
        # Make sure to brodcast to the right dimension
        ret = intersection / area1.t()
    else:
        ret = intersection / union
    return ret

def bbox_iou(bbox1, bbox2, iscrowd=False):
    ret = jaccard(bbox1, bbox2, iscrowd)
    return ret


def prep_metrics(ap_data, dets, gt, gt_masks, h, w, num_crowd):
    """ Returns a list of APs for this image, with each element being for a class  """
    gt_boxes = torch.Tensor(gt[:, :4])
    gt_boxes[:, [0, 2]] *= w
    gt_boxes[:, [1, 3]] *= h
    gt_classes = list(gt[:, 4].astype(int))
    gt_masks = torch.Tensor(gt_masks).view(-1, h*w)

    if num_crowd > 0:
        split = lambda x: (x[-num_crowd:], x[:-num_crowd])
        crowd_boxes  , gt_boxes   = split(gt_boxes)
        crowd_masks  , gt_masks   = split(gt_masks)
        crowd_classes, gt_classes = split(gt_classes)

    classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=True, score_threshold=0)

    classes = list(classes.cpu().numpy().astype(int))
    scores = list(scores.cpu().numpy().astype(float))
    masks = masks.view(-1, h*w)
    
    num_pred = len(classes)
    num_gt   = len(gt_classes)

    mask_iou_cache = mask_iou(masks, gt_masks)
    bbox_iou_cache = bbox_iou(boxes.float(), gt_boxes.float())

    if num_crowd > 0:
        crowd_mask_iou_cache = mask_iou(masks, crowd_masks, iscrowd=True)
        crowd_bbox_iou_cache = bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
    else:
        crowd_mask_iou_cache = None
        crowd_bbox_iou_cache = None

    iou_types = [
        ('box',  lambda i,j: bbox_iou_cache[i, j].item(), lambda i,j: crowd_bbox_iou_cache[i,j].item()),
        ('mask', lambda i,j: mask_iou_cache[i, j].item(), lambda i,j: crowd_mask_iou_cache[i,j].item())
    ]

    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func in iou_types:
                gt_used = [False] * len(gt_classes)

                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in range(num_pred):
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(scores[i], True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue
                                
                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(scores[i], False)


def post_process(net, dataset, file_path):
    net.detect.use_fast_nms = args.fast_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    ap_data = {
        'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
    }
    shapes = [(-1, 256, 69, 69), (-1, 256, 35, 35), (-1, 256, 18, 18), (-1, 256, 9, 9), (-1, 256, 5, 5), (-1, 138, 138, 32)]
    for it, image_idx in tqdm(enumerate(range(dataset_size))):
        img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
        all_outs = [None, None, None, None, None, None]
        for file_num in range(6):
            file_name = str(image_idx) + '_' + str(file_num) + '.bin'
            file = os.path.join(file_path, file_name)
            data = np.fromfile(file, dtype=np.float32)
            data = data.reshape(shapes[file_num])
            all_outs[file_num] = torch.from_numpy(data)
        outs = all_outs[:5]
        proto_out = all_outs[-1]
        preds = net.postprocess(outs, proto_out)
        prep_metrics(ap_data, preds, gt, gt_masks, h, w, num_crowd)
    calc_map(ap_data)


def calc_map(ap_data):
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))

    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    output_str = "\n"
    output_str += make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]) + "\n"
    output_str += make_sep(len(all_maps['box']) + 1) + "\n"
    for iou_type in ('box', 'mask'):
        output_str += make_row([iou_type] + ['%.2f' % x for x in all_maps[iou_type].values()]) + "\n"
    output_str += make_sep(len(all_maps['box']) + 1)
    print(output_str)


if __name__ == '__main__':
    parse_args()
    set_cfg(args.config)
    
    image_path = args.image_path
    json_file = args.json_file
    file_path = args.file_path
    json_path = args.ids_path

    #load dataset
    dataset = COCODetection(image_path, json_file, transform=BaseTransform(), has_gt=True)
    dataset_size = len(dataset)
    with open(json_path, 'r') as f:
        ids = json.load(f)
    dataset.ids = ids['ids']

    #load model for postprocess
    net = Yolact(training=False)
    net.load_weights(args.trained_model, args=args)

    with torch.no_grad():
        post_process(net, dataset, file_path)
