/*
 * Copyright 2022. Huawei Technologies Co., Ltd.
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

#ifndef MXBASE_Gluon_ResNet50_v1bInfer_H
#define MXBASE_Gluon_ResNet50_v1bInfer_H

#include <dirent.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "ClassPostProcessors/Resnet50PostProcess.h"

extern std::vector<double> g_inferCost;
struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t topk=5;
    std::string modelPath;
};

class Gluon_ResNet50_v1bInfer {
    public:
        APP_ERROR Init(const InitParam &initParam);
        APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);
        APP_ERROR DeInit();
        APP_ERROR BinToTensorBase(const std::string file_path, MxBase::TensorBase &tensorBase);
        APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                            std::vector<MxBase::TensorBase> &outputs);
        APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                            std::vector<std::vector<MxBase::ClassInfo>> &clsInfos);
        APP_ERROR PostProcessEval(std::vector<MxBase::TensorBase> *outputs,
                            std::vector<float> *predict);
        APP_ERROR Process(const std::string &imgFile,
                           const std::string &imgPath,
                           const std::string &outputDir_preprocess_result
                        );
        APP_ERROR WriteResult(const std::string &fileName,
                              const std::vector<float> &predict,
                              const std::string resFileName);
        APP_ERROR SaveResult(const std::string &imgFile,
                              const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos);
        APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
        APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);

    private:
        std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
        std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    //    std::shared_ptr<MxBase::ModelInferenceProcessor> post_;
        std::shared_ptr<MxBase::Resnet50PostProcess> post_;
        MxBase::ModelDesc modelDesc_;
        uint32_t deviceId_ = 0;
        // infer time
        double inferCostTimeMilliSec = 0.0;
};
#endif
