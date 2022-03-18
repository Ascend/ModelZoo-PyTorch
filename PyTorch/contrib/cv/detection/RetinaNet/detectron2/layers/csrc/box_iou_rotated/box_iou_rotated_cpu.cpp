// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// Copyright 2020 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "box_iou_rotated.h"
#include "box_iou_rotated_utils.h"

namespace detectron2 {

template <typename T>
void box_iou_rotated_cpu_kernel(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    at::Tensor& ious) {
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  for (int i = 0; i < num_boxes1; i++) {
    for (int j = 0; j < num_boxes2; j++) {
      ious[i * num_boxes2 + j] = single_box_iou_rotated<T>(
          boxes1[i].data_ptr<T>(), boxes2[j].data_ptr<T>());
    }
  }
}

at::Tensor box_iou_rotated_cpu(
    // input must be contiguous:
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);
  at::Tensor ious =
      at::empty({num_boxes1 * num_boxes2}, boxes1.options().dtype(at::kFloat));

  box_iou_rotated_cpu_kernel<float>(boxes1, boxes2, ious);

  // reshape from 1d array to 2d array
  auto shape = std::vector<int64_t>{num_boxes1, num_boxes2};
  return ious.reshape(shape);
}

} // namespace detectron2
