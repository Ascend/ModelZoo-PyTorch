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
# limitations under the License

import io
import cv2
import json
import tempfile
import contextlib
import numpy as np

from PIL import Image
from pycocotools.cocoeval import COCOeval


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)

        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"

        return format_string


class Resize(object):
    def __init__(self, fix_shape):
        self.fix_shape = fix_shape

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        long_side = w if w > h else h
        ratio = self.fix_shape / long_side
        oh = min(int(h * ratio), self.fix_shape)
        ow = min(int(ratio * w), self.fix_shape)

        return (ow, oh)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = image.resize(size, Image.BILINEAR)
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image -= self.mean
        image /= self.std

        return image


class ImgPad(object):
    def __init__(self, fix_shape):
        self.fix_shape = fix_shape

    def _pad(self, image):
        w, h, c = image.shape
        image_t = np.zeros((self.fix_shape, self.fix_shape, c), dtype=np.float32)
        image_t[0:w, 0:h, :] = image
        image_t = image_t.transpose(2, 0, 1)[None]

        return image_t

    def __call__(self, image):
        image = self._pad(image)

        return image


def build_transforms(fix_shape):
    resize = Resize(fix_shape)
    normalize_transform = Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
    pad = ImgPad(fix_shape)

    transform = Compose([resize, normalize_transform, pad])

    return transform


def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = np.zeros((N, 1, M + pad2, M + pad2), dtype=np.float32)
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]

    TO_REMOVE = 1
    w = box[2] - box[0] + TO_REMOVE
    h = box[3] - box[1] + TO_REMOVE
    w = round(max(w, 1))
    h = round(max(h, 1))

    mask = cv2.resize(mask, (w, h))

    if thresh >= 0:
        mask = np.array(mask > thresh, dtype=np.uint8)
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = (mask * 255).astype(np.uint8)

    box = [round(x) for x in box]

    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2], im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3], im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]
    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes, h, w):

        im_w, im_h = w, h
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes)
        ]
        if len(res) > 0:
            res = np.stack(res, axis=0)[:, None]
        else:
            res = np.zeros((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes, h, w):

        result = self.forward_single_image(masks, boxes, h, w)
        return result


def np_nms(boxes, scores, thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    res_keep = np.array(keep)

    return res_keep


def np_batched_nms(boxes, scores, labels, masks, iou_threshold=0.4, keep_num=100):
    if not boxes.any():
        return np.array((0,), dtype=np.int64)

    max_coordinate = boxes.max()

    offsets = np.float32(labels) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = np_nms(boxes_for_nms, scores, iou_threshold)

    if len(keep) >= keep_num:
        bboxes = boxes[keep][:keep_num]
        labels = labels[keep][:keep_num]
        scores = scores[keep][:keep_num]
        masks = masks[keep][:keep_num]
    else:
        diff_num = keep_num - len(keep)
        bboxes = np.concatenate([boxes[keep], np.zeros((diff_num, 4))], axis=0)
        labels = np.concatenate([labels[keep], np.ones(diff_num, dtype=np.int64) * -1], axis=0)
        scores = np.concatenate([scores[keep], np.zeros(diff_num, )], axis=0)
        masks = np.concatenate([masks[keep], np.zeros((diff_num, 1, 28, 28))], axis=0)

    return bboxes, labels, scores, masks


def post_process(ori_h, ori_w, ratio, bboxes, labels, scores, masks=None):
    bboxes /= ratio
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, ori_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, ori_h)

    return bboxes, labels, scores, masks


def convert_to_coco_format(coco_dataset, image_id, bboxes, labels, scores, rles):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

    data_list = []
    for i in range(len(labels)):
        label = labels[i]
        if label < 0:
            continue
        category_id = coco_dataset.dataset['categories'][label - 1]['id']
        pred_data = {
            "image_id": int(image_id),
            "category_id": category_id,
            "bbox": bboxes[i].tolist(),
            "score": float(scores[i]),
            "segmentation": rles[i],
        }  # COCO json format
        data_list.append(pred_data)

    return data_list


def evaluate_prediction(coco_dataset, data_dict):
    annType = ["segm", "bbox", "keypoints"]

    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(data_dict) > 0:
        cocoGt = coco_dataset

        _, tmp = tempfile.mkstemp()
        json.dump(data_dict, open(tmp, "w"))
        cocoDt = cocoGt.loadRes(tmp)

        coco_bbox_eval = COCOeval(cocoGt, cocoDt, annType[1])
        coco_bbox_eval.evaluate()
        coco_bbox_eval.accumulate()
        redirect_string_1 = io.StringIO()
        with contextlib.redirect_stdout(redirect_string_1):
            coco_bbox_eval.summarize()
        bbox_info = redirect_string_1.getvalue()

        coco_segm_eval = COCOeval(cocoGt, cocoDt, annType[0])
        coco_segm_eval.evaluate()
        coco_segm_eval.accumulate()
        redirect_string_2 = io.StringIO()
        with contextlib.redirect_stdout(redirect_string_2):
            coco_segm_eval.summarize()
        segm_info = redirect_string_2.getvalue()

        return bbox_info, segm_info


def draw_bbox_segm(image, bboxes, labels, scores, masks=None):
    img = image.copy()
    for i in range(len(labels)):
        box = bboxes[i]
        label = labels[i]
        class_label = label_categories[label-1]['name']
        score = scores[i]
        if masks is not None:
            mask = masks[i][0]
        if score > 0.5:
            box = list(map(int, box))
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            cv2.putText(img, f'{class_label}_{score:.3f}', (box[0], box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            if masks is not None:
                mask = (mask > 0.1).astype(np.uint8) * 255
                mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                img[y1:y2, x1:x2, 0] = img[y1:y2, x1:x2, 0] * 0.3 + mask * 0.7

    return img


label_categories = [{'supercategory': 'person', 'id': 1, 'name': 'person'},
                    {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                    {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                    {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
                    {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
                    {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
                    {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
                    {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
                    {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
                    {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
                    {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
                    {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
                    {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
                    {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
                    {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
                    {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
                    {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
                    {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
                    {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
                    {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
                    {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
                    {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
                    {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
                    {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
                    {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
                    {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
                    {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
                    {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
                    {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
                    {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
                    {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
                    {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
                    {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
                    {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
                    {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
                    {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
                    {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
                    {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
                    {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
                    {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
                    {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
                    {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
                    {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
                    {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
                    {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
                    {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
                    {'supercategory': 'food', 'id': 52, 'name': 'banana'},
                    {'supercategory': 'food', 'id': 53, 'name': 'apple'},
                    {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
                    {'supercategory': 'food', 'id': 55, 'name': 'orange'},
                    {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
                    {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
                    {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
                    {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
                    {'supercategory': 'food', 'id': 60, 'name': 'donut'},
                    {'supercategory': 'food', 'id': 61, 'name': 'cake'},
                    {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
                    {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
                    {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
                    {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
                    {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
                    {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
                    {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
                    {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
                    {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
                    {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
                    {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
                    {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
                    {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
                    {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
                    {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
                    {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
                    {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
                    {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
                    {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
                    {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
                    {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
                    {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
                    {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
                    {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]
