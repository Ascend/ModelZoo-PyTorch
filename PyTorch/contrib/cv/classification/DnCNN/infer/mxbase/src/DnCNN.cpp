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

#include "DnCNN.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <queue>
#include <utility>
#include <fstream>
#include <map>
#include <iostream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace {
    const int imageSize = 481;
}

APP_ERROR DnCNN::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    model_DnCNN = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_DnCNN->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR DnCNN::DeInit() {
    model_DnCNN->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR DnCNN::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    uint32_t dataSize =  imageSize * imageSize * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, 1, imageSize, imageSize};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR DnCNN::TensorBaseToCVMat(const MxBase::TensorBase &tensorBase, cv::Mat *imageMat) {
    uint32_t dataSize = imageSize * imageSize * sizeof(float);
    void* buffer = tensorBase.GetBuffer();
    std::vector<float> vec(imageSize * imageSize);
    memcpy(vec.data(), buffer, dataSize);
    *imageMat = cv::Mat(vec, true).reshape(0, imageSize);
    return APP_ERR_OK;
}

APP_ERROR DnCNN::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_DnCNN->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_DnCNN->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference DnCNN failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DnCNN::Process(const std::string &imgPath, float *psnr) {
    cv::Mat img_origin = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);

    cv::resize(img_origin, img_origin, cv::Size(imageSize, imageSize));

    cv::Mat img;
    img_origin.convertTo(img, CV_32F, 1.0 / 255.0);

    cv::Mat noise(imageSize, imageSize, CV_32F);
    cv::randn(noise, 0.0f, 15.0f / 255);

    cv::Mat noisy;
    cv::add(img, noise, noisy);

    std::vector<MxBase::TensorBase> inputs;
    std::vector<MxBase::TensorBase> outputs;
    MxBase::TensorBase tensorBase;
    APP_ERROR ret = CVMatToTensorBase(noisy, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret = " << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    if (!outputs[0].IsHost()) {
        outputs[0].ToHost();
    }

    cv::Mat residual;
    TensorBaseToCVMat(outputs[0], &residual);

    cv::Mat denoised;
    cv::subtract(noisy, residual, denoised);

    *psnr = cv::PSNR(img, denoised, 1.0);

    return APP_ERR_OK;
}

