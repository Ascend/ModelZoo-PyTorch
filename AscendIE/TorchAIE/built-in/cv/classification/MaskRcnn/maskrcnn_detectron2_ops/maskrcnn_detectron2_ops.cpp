/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_nms(
    at::Tensor bbox,
    at::Tensor scores,
    double score_threshold,
    double iou_threshold,
    int64_t max_size_per_class,
    int64_t max_total_size
)
{
    auto boxBatch = bbox.sizes()[0];
    auto boxFeat = bbox.sizes()[3];
    auto scoreBatch = scores.sizes()[0];

    auto outBox = torch::ones({boxBatch, max_total_size, boxFeat}).to(torch::kFloat16);
    auto outScore = torch::ones({scoreBatch, max_total_size}).to(torch::kFloat16);
    auto outClass = torch::ones({max_total_size, }).to(torch::kInt64);
    auto outNum = torch::ones({1, }).to(torch::kFloat32);

    return std::make_tuple(outBox, outScore, outClass, outNum);
}

at::Tensor roi_extractor(
    std::vector<at::Tensor> feats,
    at::Tensor rois,
    bool aligned,
    int64_t finest_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    c10::string_view pool_mode,
    int64_t roi_scale_factor,
    int64_t sample_num,
    std::vector<double> spatial_scale
)
{
    auto k = rois.sizes()[0];
    auto c = feats[0].sizes()[1];
    auto roi_feats = torch::ones({k, c, pooled_height, pooled_width}).to(torch::kFloat32);

    return roi_feats;
}

TORCH_LIBRARY(aie, m) {
    m.def("batch_nms", batch_nms);
    m.def("roi_extractor", roi_extractor);
}