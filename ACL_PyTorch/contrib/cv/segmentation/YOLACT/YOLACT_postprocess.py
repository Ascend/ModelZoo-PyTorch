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
import os
import random
import pickle
import argparse
from collections import defaultdict
from collections import OrderedDict

import torch
import pycocotools
import numpy as np

from layers import Detect
from layers.output_utils import postprocess
from layers.box_utils import jaccard, mask_iou
from data import cfg, set_cfg
from data import COCODetection, get_label_map
from utils import timer
from utils.augmentations import BaseTransform
from utils.functions import MovingAverage, ProgressBar

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--valid_images', default='/home/data/coco/images/', help='the path of validation images')
    parser.add_argument('--valid_annotations', default='/home/data/coco/annotations/instances_val2017.json', help='the path of validation annotations')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--npu_result', default=-1, type=str,
                        help='The path of npu infer result.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--cann_version', default="5", type=str,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')

    parser.set_defaults(no_bar=False, output_coco_json=False, shuffle=False,
                        no_sort=False, mask_proto_debug=False, detect=False, crop=True)

    global args
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id

def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]

class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })

def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()

def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections:Detections=None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
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

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h*w)


    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i,:],   box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i,:,:], mask_scores[i])
            return

    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt   = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item(),
             lambda i,j: crowd_bbox_iou_cache[i,j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i,j: mask_iou_cache[i, j].item(),
             lambda i,j: crowd_mask_iou_cache[i,j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)

                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
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
                        ap_obj.push(score_func(i), True)
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
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')

class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1

            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x
   
class InferResultFile():
    def __init__(self, path, fileName):
        parts = fileName.split('_')
        self.imgId = int(parts[2])
        self.outputId = int(parts[3][0])
        self.arrayValue = np.fromfile(path + fileName, dtype=np.float32)

        self.arrayDim = self.arrayValue.shape[0]
        #print('finish read file :', fileName)

def getAllFiles(path):
    allFiles = os.listdir(path)
    infoFiles = {}
    for file in allFiles:
        if '.bin' in file and 'coco_val' in file:
            infoFile = InferResultFile(path, file)
            if infoFile.imgId in infoFiles.keys():
                infoFiles[infoFile.imgId].append(infoFile)
            else:
                infoFiles[infoFile.imgId] = [infoFile]
    return infoFiles

class InferResultFileFetcher():
    def __init__(self, path):
        self.path = path

    def getInferResult(self, image_idx):
        if args.cann_version == '5':
            resultDict = {}
            for i in range(0, 4):
                fileName = 'coco_val2017_' + str(image_idx) + '_' + str(i) + '.bin'
                infoFile = InferResultFile(self.path, fileName)
                if infoFile.arrayDim == 615936:
                    resultDict[0] = infoFile
                elif infoFile.arrayDim == 1559088:
                    resultDict[1] = infoFile
                elif infoFile.arrayDim == 76992:
                    resultDict[2] = infoFile
                else:
                    resultDict[3] = infoFile
        elif args.cann_version == '6':
            resultDict = {}
            for i in range(0, 5):
                if i == 3:
                    continue
                fileName = 'coco_val2017_' + str(image_idx) + '_' + str(i) + '.bin'
                infoFile = InferResultFile(self.path, fileName)
                if infoFile.arrayDim == 615936:
                    resultDict[0] = infoFile
                elif infoFile.arrayDim == 1559088:
                    resultDict[1] = infoFile
                elif infoFile.arrayDim == 76992:
                    resultDict[2] = infoFile
                else:
                    resultDict[3] = infoFile
        return resultDict

pred_priors = None

def getPriorTensor():
    global pred_priors
    if pred_priors is None:
        from yolact import PredictionModule
        cfg._tmp_img_h = 550
        cfg._tmp_img_w = 550
        pred_priors = PredictionModule.get_YOLACT_priors().numpy()
        return pred_priors
    else:
        return pred_priors

def evaluate(path, dataset):
    cfg.mask_proto_debug = args.mask_proto_debug
    inferResultFetcher = InferResultFileFetcher(path)

    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    # For each class and iou, stores tuples (score, isPositive)
    # Index ap_data[type][iouIdx][classIdx]
    ap_data = {
        'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
    }
    detections = Detections()

    dataset_indices = list(range(len(dataset)))

    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]

    # Main eval loop
    for it, image_idx in enumerate(dataset_indices):
        timer.reset()
        with timer.env('Load Data'):
            img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
            # Test flag, do not upvote
        with timer.env('Network Extra'):
            imgId_Outputs = inferResultFetcher.getInferResult(image_idx)

            pred_mask = imgId_Outputs[0].arrayValue.reshape(1, 19248, 32)     #output1 : pred_onnx[2]
            pred_conf = imgId_Outputs[1].arrayValue.reshape(1, 19248, 81)     #output2 : pred_onnx[1]
            pred_loc = imgId_Outputs[2].arrayValue.reshape(1, 19248, 4)      #output3 : pred_onnx[0]
            pred_proto = imgId_Outputs[3].arrayValue.reshape(1, 138, 138, 32)    #output4 : pred_onnx[4]

            detect = Detect(cfg.num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
            detect.use_fast_nms = args.fast_nms
            preds = detect({'loc': torch.from_numpy(pred_loc),
                            'conf': torch.from_numpy(pred_conf),
                            'mask': torch.from_numpy(pred_mask),
                            'priors': torch.from_numpy(getPriorTensor()),  #?????
                            'proto': torch.from_numpy(pred_proto)})

        # Perform the meat of the operation here depending on our mode.

        prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)
        # First couple of images take longer because we're constructing the graph.
        # Since that's technically initialization, don't include those in the FPS calculations.
        if it > 1:
            frame_times.add(timer.total_time())
        if not args.no_bar:
            if it > 1: fps = 1 / frame_times.get_avg()
            else: fps = 0
            progress = (it+1) / dataset_size * 100
            progress_bar.set_val(it+1)
            print('\rProcessing Output Results  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                  % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')

    print('Saving data...')
    with open(args.ap_data_file, 'wb') as f:
        pickle.dump(ap_data, f)
    return calc_map(ap_data)

def calc_map(ap_data):
    print('Calculating mAP...')
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

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))

if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)
    else:
        args.config = 'yolact_base_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    #if args.image is None and args.video is None and args.images is None:
    dataset = COCODetection(args.valid_images, args.valid_annotations,
                            transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
    prep_coco_cats()

    evaluate(args.npu_result + '/', dataset)
