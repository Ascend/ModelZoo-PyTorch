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
import sys

import cv2
import numpy as np
sys.path.append('./HigherHRNet-Human-Pose-Estimation')
from lib.utils.transforms import get_affine_transform


def get_nearest_boader(value, value_list):
    value_list = sorted(value_list)
    if value_list[0] > value:
        print("warning:{}->{}".format(value, value_list[0]))
        return value_list[0]
    if value_list[-1] < value:
        print("warning:{}->{}".format(value, value_list[-1]))
        return value_list[-1]
    left = 0
    right = len(value_list)
    while left < right:
        mid = (right - left) // 2 + left
        if value_list[mid] == value:
            return value
        elif value_list[mid] < value:
            left = mid + 1
        elif value_list[mid] > value:
            right = mid

    if left + 1 < len(value_list):
        if abs(value_list[left] - value) > abs(value_list[left + 1] - value):
            return value_list[left + 1]
    return value_list[left]


def get_multi_scale_size(image, input_size, current_scale, min_scale, scale_list):
    h, w, _ = image.shape
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63) // 64 * 64)

    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        assert w_resized == 512
        h_resized_ori = int(
            int((min_input_size / w * h + 63) // 64 * 64) * current_scale / min_scale
        )
        # change h_resized to nearest value in scale_list
        h_resized = get_nearest_boader(h_resized_ori, scale_list)

        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        assert h_resized == 512
        w_resized_ori = int(
            int((min_input_size / h * w + 63) // 64 * 64) * current_scale / min_scale
        )
        # change h_resized to nearest value in scale_list
        w_resized = get_nearest_boader(w_resized_ori, scale_list)

        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def resize_align_multi_scale(image, input_size, current_scale, min_scale, scale_list):
    size_resized, center, scale = get_multi_scale_size(
        image, input_size, current_scale, min_scale, scale_list
    )
    trans = get_affine_transform(center, scale, 0, size_resized)

    image_resized = cv2.warpAffine(
        image,
        trans,
        size_resized
    )

    return image_resized, center, scale