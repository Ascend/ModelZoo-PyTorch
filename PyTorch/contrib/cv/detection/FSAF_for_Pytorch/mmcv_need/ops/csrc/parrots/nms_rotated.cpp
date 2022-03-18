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
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/nms_rotated/nms_rotated.h
#include "parrots_cpp_helper.hpp"

DArrayLite nms_rotated_cuda(const DArrayLite dets, const DArrayLite scores,
                            const DArrayLite dets_sorted, float iou_threshold,
                            const int multi_label, cudaStream_t stream,
                            CudaContext& ctx);

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
void nms_rotated(CudaContext& ctx, const SSElement& attr,
                 const OperatorBase::in_list_t& ins,
                 OperatorBase::out_list_t& outs) {
  float iou_threshold;
  int multi_label;
  SSAttrs(attr)
      .get<float>("iou_threshold", iou_threshold)
      .get<int>("multi_label", multi_label)
      .done();

  const auto& dets = ins[0];
  const auto& scores = ins[1];
  const auto& dets_sorted = ins[2];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

  outs[0] = nms_rotated_cuda(dets, scores, dets_sorted, iou_threshold,
                             multi_label, stream, ctx);
}

PARROTS_EXTENSION_REGISTER(nms_rotated)
    .attr("multi_label")
    .attr("iou_threshold")
    .input(3)
    .output(1)
    .apply(nms_rotated)
    .done();
