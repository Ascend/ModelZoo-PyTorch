/*
 * Copyright 2022 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-3.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */

#ifndef MXBASE_SHUFFLENETV2PLUSCLASSIFY_H
#define MXBASE_SHUFFLENETV2PLUSCLASSIFY_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/postprocess/include/ClassPostProcessors/Resnet50PostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

#define CHANNEL 3
#define NET_HEIGHT 224
#define NET_WIDTH 224
extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    uint32_t topk;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

class Shufflenetv2plusClassify {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadTensorFromFile(const std::string &file, float *data);
    APP_ERROR ReadInputTensor(const std::string &fileName, std::vector<MxBase::TensorBase> *inputs);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
        std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
        std::vector<std::vector<MxBase::ClassInfo>> &clsInfos);
    APP_ERROR Process(const std::string &imgPath);
    APP_ERROR SaveResult(const std::string &imgPath,
        std::vector<std::vector<MxBase::ClassInfo>> &BatchClsInfos);
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::Resnet50PostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    std::ofstream tfile_;

    std::vector<uint32_t> inputDataShape_ = {1, 3, 224, 224};
    uint32_t inputDataSize_ = 50000;
};
#endif