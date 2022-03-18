/*
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FCOS_H
#define FCOS_H

#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class FCOS {
public:

    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat, int& height, int& width);
    APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
    APP_ERROR VectorToTensorBase(int* transMat, MxBase::TensorBase& tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::string& imgPath, std::vector<MxBase::TensorBase> &inputs, 
                          const std::string &subresultPath,int& height, int& width,const std::string& name,
                          std::string &showPath,float& PROB_THRES);
    APP_ERROR Process(const std::string &dirPath,  std::string &resultPath,std::string &showPath,float& PROB_THRES);

private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::vector<std::string> GetFileList(const std::string &dirPath);
    MxBase::ModelDesc modelDesc_;
    const int device_id = 0;
    uint32_t deviceId_ = device_id;
};
#endif
