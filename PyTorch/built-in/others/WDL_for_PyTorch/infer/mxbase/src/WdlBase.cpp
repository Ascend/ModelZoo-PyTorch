/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "WdlBase.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

const uint32_t FEATURE_LENGTH = 39;

APP_ERROR WdlBase::Init(const InitParam &initParam) {
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

APP_ERROR WdlBase::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR WdlBase::ReadInputTensor(float *data, uint32_t index,
                                       std::vector<MxBase::TensorBase> *inputs) {
    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, FEATURE_LENGTH};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT32));
    return APP_ERR_OK;
}

APP_ERROR WdlBase::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                 std::vector<MxBase::TensorBase> *outputs) {
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
        outputs->push_back(tensor);
    }

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR WdlBase::PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<float> *inferResults) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // output output tensor shape and infer result
    auto outputShape = tensor.GetShape();
    LogInfo << "==============================================================";
    LogInfo << "output shape is: " << outputShape[0] << std::endl;
    void* data = tensor.GetBuffer();
    float value = *(reinterpret_cast<float*>(data));
    LogInfo << "infer result is: " << value << std::endl;
    LogInfo << "==============================================================";
    inferResults->push_back(value);
    return APP_ERR_OK;
}

APP_ERROR WdlBase::WriteInferResultAndLabel(const std::string &fileName, const std::vector<float> &inferResults,
                                            const std::vector<uint32_t> &groundTruths) {
    std::string resultPathName = "result";
    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }

    // create label file under result directory
    std::string groundTruthPathName = resultPathName + "/ground_truth.txt";
    std::ofstream groundTruthOutFile(groundTruthPathName, std::ofstream::out);
    if (groundTruthOutFile.fail()) {
        LogError << "Failed to open label file: " << groundTruthPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // write label into file
    LogInfo << "data label in " << groundTruthPathName << std::endl;

    for (auto &groundTruth : groundTruths) {
        groundTruthOutFile << groundTruth << std::endl;
    }
    groundTruthOutFile.close();

    // create inference result file under result directory
    std::string inferPathName = resultPathName + "/infer_result.txt";
    std::ofstream inferOutFile(inferPathName, std::ofstream::out);
    if (inferOutFile.fail()) {
        LogError << "Failed to open infer result file: " << inferPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // write inference result into file
    LogInfo << "infer result in " << inferPathName << std::endl;

    for (auto &inferResult : inferResults) {
        inferOutFile << inferResult << std::endl;
    }
    inferOutFile.close();
    return APP_ERR_OK;
}

APP_ERROR WdlBase::Process(const std::string &inferPath, const std::string &fileName) {
    std::string inputDataFile = inferPath + fileName;

    std::vector<uint32_t> groundTruths = {};
    std::vector<float> inferResults = {};

    // read data file(csv data format)
    std::ifstream inFile(inputDataFile);
    std::string line;
    // drop the first row - labels
    std::getline(inFile, line);
    while (std::getline(inFile, line)) {
        std::istringstream strIn(line);
        float data[FEATURE_LENGTH] = {0.0};
        std::string field;
        std::getline(strIn, field, ',');
        groundTruths.push_back(std::stoi(field));
        uint32_t idx = 0;
        while (std::getline(strIn, field, ',')) {
            data[idx] = std::stof(field);
            idx++;
        }

        std::vector<MxBase::TensorBase> inputs = {};
        APP_ERROR ret = ReadInputTensor(data, INPUT_DATAS, &inputs);
        if (ret != APP_ERR_OK) {
            LogError << "Read data failed, ret=" << ret << ".";
            return ret;
        }

        std::vector<MxBase::TensorBase> outputs = {};
        ret = Inference(inputs, &outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }

        ret = PostProcess(&outputs, &inferResults);
        if (ret != APP_ERR_OK) {
            LogError << "PostProcess failed, ret=" << ret << ".";
            return ret;
        }
    }

    APP_ERROR ret = WriteInferResultAndLabel(fileName, inferResults, groundTruths);
    if (ret != APP_ERR_OK) {
        LogError << "save infer result and label failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}
