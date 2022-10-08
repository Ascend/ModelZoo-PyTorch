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

#ifndef MXBASE_UNet_H
#define MXBASE_UNet_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
    std::string outputDataPath;
    std::string outputBinPath;
};

class UNet {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &inferPath, const std::string &fileName);
    APP_ERROR SaveInferResult(const std::vector<MxBase::TensorBase> &outputs);

 protected:
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase);
    APP_ERROR BinWriteResult(const std::string &imageFile, std::vector<MxBase::TensorBase> outputs);
    APP_ERROR ImageWriteResult(MxBase::TensorBase *tensor,cv::Mat &output);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    std::string outputDataPath_ = "./result";
    std::string outputBinPath_ = "./bin_result";
    std::vector<uint32_t> inputDataShape_ = {1,3,96,96};
    uint32_t inputDataSize_ = 27648;
};

#endif