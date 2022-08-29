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

#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <map>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include "DeepMarClassify.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR DeepMarClassify::Init(const InitParam &initParam) {
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

    uint32_t input_data_size = 1;
    for (size_t j = 0; j < this->modelDesc_.inputTensors[0].tensorDims.size(); ++j) {
        this->inputDataShape_[j] = (uint32_t)this->modelDesc_.inputTensors[0].tensorDims[j];
        input_data_size *= this->inputDataShape_[j];
    }
    this->inputDataSize_ = input_data_size;

    MxBase::ConfigData configData;
    const std::string softmax = initParam.softmax ? "true" : "false";
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("TOP_K", std::to_string(initParam.topk));
    configData.SetJsonValue("SOFTMAX", softmax);
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    tfile_.open("mx_pred_result.txt");
    if (!tfile_) {
        LogError << "Open result file failed.";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    return APP_ERR_OK;
}

APP_ERROR DeepMarClassify::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    tfile_.close();
    return APP_ERR_OK;
}

APP_ERROR DeepMarClassify::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DeepMarClassify::ReadTensorFromFile(const std::string &file, float *data) {
    if (data == NULL) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    std::ifstream infile;
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    if (infile.fail()) {
        LogError << "Failed to open data file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(data), sizeof(float) * this->inputDataSize_);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR DeepMarClassify::ReadInputTensor(const std::string &fileName, std::vector<MxBase::TensorBase> *inputs) {
    float data[this->inputDataSize_] = {0};
    APP_ERROR ret = ReadTensorFromFile(fileName, data);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }
    const uint32_t dataSize = this->modelDesc_.inputTensors[0].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, this->deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, this->inputDataShape_, MxBase::TENSOR_DTYPE_FLOAT32));
    return APP_ERR_OK;
}

APP_ERROR DeepMarClassify::Process(const std::string &imgPath) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputIdsFile = imgPath;
    APP_ERROR ret = ReadInputTensor(inputIdsFile, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input ids failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> outputs = {};

    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    if (!outputs[0].IsHost()){
        outputs[0].ToHost();
    }
    float* value = (float*)((void*)outputs[0].GetBuffer());
    uint32_t idx = 0;
    float v = *(value + idx);
    std::string data;
    while (v) {
        data += std::to_string(v) + " ";
        idx++;
        if(idx % 5 == 0){
            data += "\n";
        }
        v = *(value + idx);
    }
    LogInfo << "The data in the output TensorBase is: " << std::endl << data;
    tfile_ << imgPath.substr(10, 5) << ": \n" << data << std::endl;
    return APP_ERR_OK;
}
