# coding: utf-8
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

import argparse
import json
import os
from contextlib import ExitStack

import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType

from biz import dto
from biz.predictor import AscendPredictor


def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path.rstrip('/')))[0]


def _calc_resize_keep_aspect_ratio_short(size, resize):
    w, h = size
    if w > h:
        oh = resize
        ow = int(1.0 * w * oh / h)
    else:
        ow = resize
        oh = int(1.0 * h * ow / w)
    return ow, oh


def _calc_fix_scale_crop_coordinates(pil_img, crop_size):
    ow, oh = _calc_resize_keep_aspect_ratio_short(pil_img.size,
                                                  crop_size)
    x1 = int(round((ow - crop_size) / 2.))
    y1 = int(round((oh - crop_size) / 2.))
    crop_coord= x1, y1, x1 + crop_size, y1 + crop_size
    print(f" {pil_img.size} to crop {crop_coord}")
    return crop_coord


def calc_fix_scale_crop_coordinates(img_path, crop_size):
    print(f"calc crop {img_path}")
    pil_img = Image.open(img_path).convert("RGB")
    return _calc_fix_scale_crop_coordinates(pil_img, crop_size)


class VOCSegmentationDataLoader(object):
    def __init__(
        self,
        base_dir,
        split='val',
        num_classes=21,
        crop_size=513,
        limit=None
    ):
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        self.crop_size = crop_size
        self.num_clsses = num_classes
        self.split = split

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        with open(os.path.join(_splits_dir, f"{split}.txt"), 'r') as fd:
            lines = fd.read().splitlines()

        self.im_ids = []
        self.images = []
        self.categories = []
        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, f"{line}.jpg")
            _cat = os.path.join(self._cat_dir, f"{line}.png")

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        total = len(self.im_ids)
        self.limit = total if limit is None else min(total, limit)
        self.cur_index = 0

    def _make_img_gt_point_pair(self, index):
        with open(self.images[index], 'rb') as fd:
            img = fd.read()
        target = Image.open(self.categories[index])
        return img, target

    def transform_target(self, img, crop_size):
        resize = _calc_resize_keep_aspect_ratio_short(img.size, crop_size)
        img = img.resize(resize)
        crop_coord = _calc_fix_scale_crop_coordinates(img, crop_size)
        img = img.crop(crop_coord)
        return np.array(img).astype(np.float32)

    def __iter__(self):
        return self

    def __next__(self) -> dto.DataItem:
        if self.cur_index == self.limit:
            raise StopIteration()

        file_path = self.images[self.cur_index]
        img, target = self._make_img_gt_point_pair(self.cur_index)

        self.cur_index += 1

        return dto.DataItem(**{
            'file_path': file_path,
            'file_name': get_file_name(self.images[self.cur_index - 1]),
            'img': img,
            'gt': self.transform_target(target, self.crop_size),
            'crop_size': self.crop_size,
        })


class Predictor(AscendPredictor):
    def __init__(self, pipeline_conf, stream_name):
        super().__init__(pipeline_conf, stream_name)

    def _predict(self, data_item: dto.DataItem):
        plugin_0 = 0
        plugin_1 = 1

        crop = calc_fix_scale_crop_coordinates(data_item.file_path,
                                               data_item.crop_size)
        object_list = MxpiDataType.MxpiObjectList()
        obj_vec = object_list.objectVec.add()
        obj_vec.x0, obj_vec.y0, obj_vec.x1, obj_vec.y1 = crop
        protobuf_data = self.create_protobuf_data(b'appsrc1',
                                                  b'MxTools.MxpiObjectList',
                                                  object_list)
        # send crop coordinates
        self.send_protobuf_data(self.stream_name, plugin_1, protobuf_data)

        # send image data
        data_input = self.create_data(data_item.img)
        unique_id = self.send_data(self.stream_name, plugin_0, data_input)

        # get result
        result = self.get_result(self.stream_name, unique_id)
        return json.loads(result.data.decode())

    def post_process(self, pred) -> dto.PredictResult:
        mx_image_mask = dto.MxpiImageMask.from_dict(pred)
        img_mask = mx_image_mask.MxpiImageMask[0]
        return dto.PredictResult(mask=img_mask.mask)


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def mean_intersection_over_union(self):
        iou = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        miou = np.nanmean(iou)
        return miou

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def calc_acc(mask, pred, num_classes):
    def _cal_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np\
            .bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2)\
            .reshape(n, n)

    return _cal_hist(mask.flatten(), pred.flatten(), num_classes)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', help='voc data dir.')
    parser.add_argument('result_dir', help='result file')
    return parser.parse_args()


def main():
    pipeline_conf = "./etc/deeplabv3plus_opencv.pipeline.json"
    stream_name = b'segmentation'
    crop_size = 513
    num_classes = 21

    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True, mode=0o750)
    dataset = VOCSegmentationDataLoader(args.data_dir,
                                        num_classes=num_classes,
                                        crop_size=crop_size,
                                        limit=None)
    with ExitStack() as stack:
        predictor = stack.enter_context(Predictor(pipeline_conf, stream_name))
        evaluator = Evaluator(num_classes)

        for data_item in dataset:
            print(f"pred {data_item.file_name}")
            pred_result = predictor.predict(data_item)
            pred_result.save_to_png(args.result_dir, data_item.file_name)
            evaluator.add_batch(data_item.gt.flatten().astype(int),
                                pred_result.mask.flatten())

    miou = evaluator.mean_intersection_over_union()
    print("mean IoU", miou)


if __name__ == "__main__":
    main()
