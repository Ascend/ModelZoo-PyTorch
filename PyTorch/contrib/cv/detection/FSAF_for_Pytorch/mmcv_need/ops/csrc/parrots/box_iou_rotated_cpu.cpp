# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cpu.cpp
#include "box_iou_rotated_utils.hpp"
#include "parrots_cpp_helper.hpp"

template <typename T>
void box_iou_rotated_cpu_kernel(const DArrayLite boxes1,
                                const DArrayLite boxes2, DArrayLite ious,
                                const int mode_flag, const bool aligned) {
  int output_size = ious.size();
  int num_boxes1 = boxes1.dim(0);
  int num_boxes2 = boxes2.dim(0);

  auto ious_ptr = ious.ptr<float>();

  if (aligned) {
    for (int i = 0; i < output_size; i++) {
      ious_ptr[i] = single_box_iou_rotated<T>(boxes1[i].ptr<T>(),
                                              boxes2[i].ptr<T>(), mode_flag);
    }
  } else {
    for (int i = 0; i < num_boxes1; i++) {
      for (int j = 0; j < num_boxes2; j++) {
        ious_ptr[i * num_boxes2 + j] = single_box_iou_rotated<T>(
            boxes1[i].ptr<T>(), boxes2[j].ptr<T>(), mode_flag);
      }
    }
  }
}

void box_iou_rotated_cpu_launcher(const DArrayLite boxes1,
                                  const DArrayLite boxes2, DArrayLite ious,
                                  const int mode_flag, const bool aligned) {
  box_iou_rotated_cpu_kernel<float>(boxes1, boxes2, ious, mode_flag, aligned);
}
