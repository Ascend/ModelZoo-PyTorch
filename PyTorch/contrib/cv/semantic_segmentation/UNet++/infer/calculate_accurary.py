# coding=utf-8
#
# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import argparse
import numpy as np
import cv2 as cv
from glob import glob


class MultiClassLoader:
    _plugin_name = "multiclass"

    def __init__(self, dataset_dir):
        super(MultiClassLoader, self).__init__()
        self._dataset_dir = dataset_dir

    def iter_dataset(self):
        for image_id in self.list_image_id():
            image, mask = self.get_image_mask(image_id)
            yield image_id, image, mask

    def list_image_id(self):
        img_ids = glob(os.path.join(self._dataset_dir, 'images', '*.png'))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        return img_ids
        
    def get_image_mask(self, image_id):
        image_dir = os.path.join(self._dataset_dir, 'images')
        image = cv.imread(os.path.join(image_dir, image_id+".png"))
        mask_dir = os.path.join(self._dataset_dir, 'masks/0')
        mask = cv.imread(os.path.join(mask_dir, image_id+".png"), cv.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to get image by id {image_id}")
            
        mask = mask.astype('float32') / 255

        return image, mask

    @classmethod
    def get_plugin_name(cls):
        return cls._plugin_name


def _parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="path of multiclass")
    parser.add_argument("--infer_dir", type=str, required=True,
                        help="path of infer npy result")
    return parser.parse_args()

def sigmoid(x):
    y = x.copy()
    y[x >= 0] = 1.0 / (1 + np.exp(-x[x >= 0]))
    y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
    return y

def _calculate_accuracy(infer_image, mask_image):
    """
    calculate iou

    Args:
        infer_image (array): The result of an inference, of type array
        mask_image (array): The label of an image, of type array

    Returns:
        float

    """
    smooth = 1e-5
    
    infer_image = infer_image.astype('float32') / 255.0
    infer_image_ =  infer_image > 0.5
    mask_image_ = mask_image > 0.5

    inter = (infer_image_ & mask_image_).sum()
    union = (infer_image_ | mask_image_).sum()

    single_iou = (inter + smooth) / (union + smooth)
    return single_iou

def calculate_origin_accuracy(multiclass_dir, infer_result_dir):
    data_loader = MultiClassLoader(multiclass_dir)
    iou_sum = 0.0
    cnt = 0
    for image_id, _, mask in data_loader.iter_dataset():
        infer = cv.imread(os.path.join(infer_result_dir, f"{image_id}.png"), cv.IMREAD_GRAYSCALE)
        iou = _calculate_accuracy(infer, mask)
        print(f"{image_id} single iou is {iou}")
        iou_sum += iou
        cnt += 1

    print(f"========== Cross Valid IOU is: {iou_sum / cnt}")


if __name__ == '__main__':
    args = _parser_args()
    calculate_origin_accuracy(args.dataset_dir, args.infer_dir)
