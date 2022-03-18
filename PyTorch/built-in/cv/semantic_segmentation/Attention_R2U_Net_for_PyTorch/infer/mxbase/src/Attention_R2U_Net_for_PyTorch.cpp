/*
 * Copyright 2021 Huawei Technologies Co., Ltd
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ou may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */

#include "Attention_R2U_Net_for_PyTorch.h"
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
#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

#include <typeinfo>

using namespace MxBase;

constexpr auto model_name = "Attention_R2U_Net";

APP_ERROR Attention_R2U_Net::Init(const InitParam &initParam) {
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
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Attention_R2U_Net::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Attention_R2U_Net::VectorToTensorBase_float(const std::vector<float> &test_image_vector,
                                                      MxBase::TensorBase &tensorBase) {
    uint32_t dataSize_test_image = 1;
    std::vector<uint32_t> shape_test_image = {1, 3, 224, 224};

    for (uint32_t i = 0; i < shape_test_image.size(); i++) {
            dataSize_test_image *= shape_test_image[i];
    }
    float *metaData_test_image = new float[dataSize_test_image];

    uint32_t idx = 0;
    for (size_t i = 0; i < test_image_vector.size(); i++) {
        metaData_test_image[idx++] = test_image_vector[i];
    }
    MemoryData memoryDataDst(dataSize_test_image*4, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(metaData_test_image, dataSize_test_image*4, MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    tensorBase = TensorBase(memoryDataDst, false, shape_test_image, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}


APP_ERROR Attention_R2U_Net::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                       std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
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
        outputs.push_back(tensor);
    }

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    return APP_ERR_OK;
}


APP_ERROR Attention_R2U_Net::SaveInferResult(const std::vector<MxBase::TensorBase> &outputs) {
    uint32_t batchIndex = 0;
    auto retTensor = outputs[0];
    std::vector<uint32_t> indices = {};
    std::vector<uint32_t> shape = retTensor.GetShape();
    uint32_t size = 1*1*224*224;
    for (uint32_t i = 0; i < size; i++) {
        indices.push_back(i);
    }
    if (!retTensor.IsHost()) {
        retTensor.ToHost();
    }
    float *value = (float *)retTensor.GetBuffer();
    std::string resFileName = "./res.txt";
    std::ofstream outfile(resFileName, std::ios::app);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    std::string resultStr;
    for (uint32_t i = 0; i < size-1; i++) {
        float v = *(value + i);
        resultStr += std::to_string(v) + " ";
    }
    resultStr += std::to_string(size-1) + " ";
    outfile << resultStr << "\n";
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR Attention_R2U_Net::Process(const std::vector<float> &test_image) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;

    APP_ERROR ret = VectorToTensorBase_float(test_image, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "VectorToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveInferResult(outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
