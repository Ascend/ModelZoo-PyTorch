/*
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOURGLASS_POSTPROCESS_H
#define HOURGLASS_POSTPROCESS_H

#include <vector>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"


namespace {
    auto floatDeleter = [](float* p) {};
    const int SCALE_RATIO = 200;
    const int NPOINTS = 16;  
    const int HEIGHT_HEAPMAP = 96;
    const int WIDTH_HEAPMAP = 96;
    const int NUMS_HEAPMAP = HEIGHT_HEAPMAP * WIDTH_HEAPMAP;
    float heatmaps_reshape[NPOINTS][NUMS_HEAPMAP] = {};
    float batch_heatmaps[NPOINTS][HEIGHT_HEAPMAP][WIDTH_HEAPMAP] = {};
}


class HourglassPostprocess : public MxBase::ObjectPostProcessBase {
public:
    APP_ERROR Init();
    APP_ERROR DeInit();
    APP_ERROR Process(const float center[], const float scale[],
                        const std::vector<MxBase::TensorBase> &tensors,
                        std::vector<std::vector<float> >* node_score_list);
private:
    void GetHeatmap(const std::vector<MxBase::TensorBase>& tensors,
                    uint32_t heatmapHeight, uint32_t heatmapWeight);
    int GetIntData(const int index, const float(*heatmaps_reshape)[NUMS_HEAPMAP]);
    double GetFloatData(const int index, const float(*heatmaps_reshape)[NUMS_HEAPMAP]);
    void GetAffineMatrix(const float center[], const float scale[], cv::Mat *warp_mat);
    void ParseHeatmap(const std::vector<MxBase::TensorBase>& tensors,
                        std::vector<float> *preds_result,
                        uint32_t heatmapHeight, uint32_t heatmapWeight,
                        const float center[], const float scale[]);
};

#endif
