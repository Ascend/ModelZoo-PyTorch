/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef GHOSTNET_CLASSIFY_H
#define GHOSTNET_CLASSIFY_H

#include "opencv2/opencv.hpp"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "ClassPostProcessors/Resnet50PostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    uint32_t topk;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

class GhostNetClassify {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imgMat);
    APP_ERROR Resize(const cv::Mat &srcMat, cv::Mat &dstMat);
    APP_ERROR CvMatToTensorBase(const cv::Mat &imgMat, MxBase::TensorBase &tensorBase);
    APP_ERROR Crop(const cv::Mat &srcMat, cv::Mat &dstMat);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
        std::vector<std::vector<MxBase::ClassInfo>> &clsInfos);
    APP_ERROR Process(const std::string &imgPath);
    double GetInferCostTimeMs() const {
        return inferCostTimeMs_;
    }

private:
    std::string FormatResultFile(const std::string &imgPath);
    APP_ERROR SaveResult(const std::string &imgPath, const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos);
private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::Resnet50PostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    double inferCostTimeMs_ = 0.0;
};
#endif
