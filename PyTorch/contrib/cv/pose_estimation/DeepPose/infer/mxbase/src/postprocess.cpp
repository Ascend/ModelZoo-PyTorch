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
 
#include "postprocess.h"
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>

APP_ERROR DeepPosePostprocess::Init() {
    LogInfo << "Begin to initialize DeepPosePostprocess.";
    return APP_ERR_OK;
}

APP_ERROR DeepPosePostprocess::DeInit() {
    LogInfo << "Begin to deinitialize DeepPosePostprocess.";
    return APP_ERR_OK;
}

void DeepPosePostprocess::FliplrRegression(const std::vector<MxBase::TensorBase>& tensors) {
    GetHeatmap(tensors, heatmaps_reshape_flip);
    float x_c = 0.5;
    
    for(int i = 0; i < 8; i++){
        int left = FLIPS_PAIRS[i][0];
        int right = FLIPS_PAIRS[i][1];
        for(int j = 0; j< COORD_DIM; j++)
        {
            float tmp = heatmaps_reshape_flip[left][j];
            heatmaps_reshape_flip[left][j] = heatmaps_reshape_flip[right][j];
            heatmaps_reshape_flip[right][j] = tmp;

        }
    }
    for(int i=0; i < NPOINTS; i++){
        heatmaps_reshape_flip[i][0] = x_c*2- heatmaps_reshape_flip[i][0];
    }
}


void DeepPosePostprocess::GetHeatmap(const std::vector<MxBase::TensorBase>& tensors, float heatmaps[NPOINTS][COORD_DIM]) {
    auto bboxPtr = reinterpret_cast<float*>(tensors[0].GetBuffer());
    std::shared_ptr<void> keypoint_pointer;
    keypoint_pointer.reset(bboxPtr, floatDeleter);

    for (size_t i = 0; i < NPOINTS; i++) {
        int startIndex = i * COORD_DIM;
        for (size_t j = 0; j < COORD_DIM; j++) {
            float x = static_cast<float*>(keypoint_pointer.get())[startIndex + j];
            heatmaps[i][j] =x;
        }
    }
}

void DeepPosePostprocess::ParseHeatmap(const std::vector<MxBase::TensorBase>& tensors,
                                        const std::vector<MxBase::TensorBase>& tensors_flip,
                                        std::vector<float> *preds_result,
                                        const float center[], const float scale[],
                                        const ImageAnnot &imageAnnot) {
    LogInfo << "Begin to ParseHeatmap.";
    GetHeatmap(tensors, heatmaps_reshape);
    FliplrRegression(tensors_flip);
    for(int i=0 ; i < NPOINTS; i++){
        for(int j = 0; j < COORD_DIM; j++){
           
            heatmaps_reshape[i][j] =  (heatmaps_reshape[i][j] + heatmaps_reshape_flip[i][j]) *0.5;
        }
    }
    void *tmp = heatmaps_reshape;
    
    std::string outputImagePath = "./result_bin/";
    std::string suffix= "._bin";

    FILE *outputFile_ = fopen((outputImagePath + imageAnnot.bboxID +imageAnnot.imageName + suffix).c_str(), "wb");
    fwrite(tmp, NPOINTS*COORD_DIM, sizeof(float), outputFile_);
    fclose(outputFile_);

    for(int i=0 ; i < NPOINTS; i++){
        for(int j = 0; j < COORD_DIM; j++){
            heatmaps_reshape[i][j] = heatmaps_reshape[i][j]* image_size[j];
        }
    }

    float maxvals[NPOINTS] = {};
    for(int i=0; i < NPOINTS; i++) maxvals[i] = 1;
    float scalex_tmp = scale[0] * SCALE_RATIO;
    float scaley_tmp  = scale[1] * SCALE_RATIO;

    float scale_x = scalex_tmp / image_size[0];
    float scale_y = scaley_tmp / image_size[1];

    for(int i = 0; i < NPOINTS; i++){
        float x_coord = heatmaps_reshape[i][0] * scale_x + center[0] - scalex_tmp * 0.5;
        float y_coord = heatmaps_reshape[i][1] * scale_y + center[1] - scaley_tmp * 0.5;
        preds_result->push_back(x_coord);
        preds_result->push_back(y_coord);
        preds_result->push_back(maxvals[i]);
    }
}

APP_ERROR DeepPosePostprocess::Process(const float center[], const float scale[],
                                        const std::vector<MxBase::TensorBase> &tensors,
                                        const std::vector<MxBase::TensorBase> &tensors_flip,
                                        std::vector<std::vector<float>>* node_score_list,
                                        const ImageAnnot &imageAnnot
                                        ) {
    LogDebug << "Begin to DeepPose PostProcess.";
    auto inputs = tensors;
    APP_ERROR ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed, ret=" << ret;
        return ret;
    }

    auto inputs_flip = tensors_flip;
    ret = CheckAndMoveTensors(inputs_flip);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed, ret=" << ret;
        return ret;
    }
       
    auto shape = inputs[0].GetShape();
    uint32_t batchSize = shape[0];
    uint32_t numKeyPoints = NPOINTS;
    uint32_t dim = COORD_DIM;
    for (uint32_t i = 0; i < batchSize; ++i) {
        std::vector<float> preds_result;
        ParseHeatmap(inputs, inputs_flip, &preds_result, center, scale, imageAnnot);
        node_score_list->push_back(preds_result);
    }    
    LogInfo << "End to DeepPose PostProcess.";
    return APP_ERR_OK;
}

