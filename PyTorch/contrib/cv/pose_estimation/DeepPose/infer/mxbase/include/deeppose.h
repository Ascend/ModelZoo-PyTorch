/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef DEEPPOSE_H
#define DEEPPOSE_H

#include <vector>
#include <string>
#include "postprocess.h"
#include <opencv2/opencv.hpp>
#include "MxBase/Log/Log.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/DeviceManager/DeviceManager.h"


namespace {
    const uint32_t MODEL_HEIGHT = 256;
    const uint32_t MODEL_WIDTH = 192;
    const uint32_t NUMS_JOINTS = 17;
    const uint32_t DIM = 2;
    bool IS_SAVED_IMGAE = false;
}

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
    std::string csvPath;
    std::string imagePath;
};

struct ImageShape {
    uint32_t width;
    uint32_t height;
};

class DeepPose {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat, ImageShape &imgShape);
    APP_ERROR Resize_Affine(const cv::Mat& srcImage, cv::Mat *dstImage, ImageShape *imgShape, 
                            const float center[], const float scale[]);
    APP_ERROR CVMatToTensorBase(const cv::Mat& imageMat, MxBase::TensorBase *tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                        std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                            const std::vector<MxBase::TensorBase> &inputs_flip,
                            std::vector<std::vector<float> >* node_score_list,
                            const float center[], const float scale[],
                            const ImageAnnot &imageAnnot);	
    APP_ERROR Process(const ImageAnnot &imageAnnot, std::string imagePath);
    APP_ERROR GetOutputs(const cv::Mat& imageMat, std::vector<MxBase::TensorBase> *outputs);
private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    std::shared_ptr<DeepPosePostprocess> deepPosePostprocess;
    uint32_t deviceId_ = 0;
};
#endif
