// Copyright 2021 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the License);
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
//

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated.h
#include "pytorch_cpp_helper.hpp"

void box_iou_rotated_cpu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned);

#ifdef MMCV_WITH_CUDA
void box_iou_rotated_cuda(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
void box_iou_rotated(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                     const int mode_flag, const bool aligned) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    box_iou_rotated_cuda(boxes1, boxes2, ious, mode_flag, aligned);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    box_iou_rotated_cpu(boxes1, boxes2, ious, mode_flag, aligned);
  }
}
