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

#ifndef DEEPPOSE_POSTPROCESS_H
#define DEEPPOSE_POSTPROCESS_H

#include <vector>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace {
    auto floatDeleter = [](float* p) {};
    const int SCALE_RATIO = 200;
    const int NPOINTS = 17;  
    const int COORD_DIM = 2;
    const int FLIPS_PAIRS[8][2] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}};
    float heatmaps_reshape[NPOINTS][COORD_DIM] = {};
    float heatmaps_reshape_flip[NPOINTS][COORD_DIM] = {};
    int image_size[COORD_DIM] = {192, 256};
}

struct ImageAnnot {
    std::string imageName;
    float center[2];
    float scale[2];
    std::string bboxID;
};

class DeepPosePostprocess : public MxBase::ObjectPostProcessBase {
public:
    APP_ERROR Init();
    APP_ERROR DeInit();
    APP_ERROR Process(const float center[], const float scale[],
                        const std::vector<MxBase::TensorBase> &tensors,
                        const std::vector<MxBase::TensorBase> &tensors_flip,
                        std::vector<std::vector<float> >* node_score_list,
                        const ImageAnnot &imageAnnot);
private:
    void GetHeatmap(const std::vector<MxBase::TensorBase>& tensors, float heatmaps[NPOINTS][COORD_DIM]);
   
    void ParseHeatmap(const std::vector<MxBase::TensorBase>& tensors,
                        const std::vector<MxBase::TensorBase>& tensors_flip,
                        std::vector<float> *preds_result,
                        const float center[], const float scale[],
                        const ImageAnnot &imageAnnot);
    void FliplrRegression(const std::vector<MxBase::TensorBase>& tensors);
};

#endif
