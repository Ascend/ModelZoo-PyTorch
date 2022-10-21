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
#include "InceptionResnetV2.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iomanip>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include <opencv2/dnn.hpp>

using namespace MxBase;

APP_ERROR InceptionResnetV2Infer::Init(const InitParam &initParam)
{
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK)
    {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK)
    {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK)
    {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::ConfigData configData;
    const std::string softmax = initParam.softmax ? "true" : "false";
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("TOP_K", std::to_string(initParam.topk));
    configData.SetJsonValue("SOFTMAX", softmax);
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::Resnet50PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK)
    {
        LogError << "Resnet50PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Infer::BinToTensorBase(const std::string file_path, MxBase::TensorBase &tensorBase)
{
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i)
    {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j)
        {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s)
        {
            dataSize *= shape[s];
        }
    }
    float *img_data = new float[dataSize];
    std::ifstream inF(file_path, std::ios::binary);
    inF.read(reinterpret_cast<char *>(img_data), sizeof(float) * (dataSize));
    inF.close();
    MxBase::MemoryData memoryDataDst(dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(&img_data[0]), dataSize * 4,
                                     MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK)
    {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, 3, 299, 299};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    delete[] img_data;
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Infer::DeInit()
{
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Infer::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                            std::vector<MxBase::TensorBase> &outputs)
{
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i)
    {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j)
        {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK)
        {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK)
    {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Infer::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                              std::vector<std::vector<MxBase::ClassInfo>> &clsInfos)
{
    APP_ERROR ret = post_->Process(inputs, clsInfos);
    if (ret != APP_ERR_OK)
    {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Infer::SaveResult(const std::string &outputPath, const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos, const std::string name)
{
    // write inference result into file
    LogInfo << "==============================================================";
    LogInfo << "Infer finished!";
    // Determine whether the outputPath folder exists, if it does not exist, create it
    if (access(outputPath.c_str(), F_OK) == -1)
    {
        mkdir(outputPath.c_str(), S_IRWXO | S_IRWXG | S_IRWXU);
    }
    for (auto clsInfos : batchClsInfos)
    {
        // create result file
        std::string resFileName = outputPath + name + "_1.txt";
        std::cout << "resFileName" << resFileName << std::endl;
        std::ofstream tfile(resFileName, std::ofstream::app);
        if (tfile.fail())
        {
            LogError << "Failed to open result file: " << outputPath;
            return APP_ERR_COMM_OPEN_FAIL;
        }
        std::string resultStr;
        for (auto clsInfo : clsInfos)
        {
            LogDebug << " className:" << clsInfo.className << " confidence:" << clsInfo.confidence << " classIndex:" << clsInfo.classId;
            resultStr += std::to_string(clsInfo.classId) + " ";
        }
        tfile << resultStr << std::endl;
        tfile.close();
    }
    LogInfo << "==============================================================";
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Infer::Process(const std::string &imgPath, const std::string &outputPath)
{
    MxBase::TensorBase tensorBase;
    APP_ERROR ret = BinToTensorBase(imgPath, tensorBase);
    if (ret != APP_ERR_OK)
    {
        LogError << "BinToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    // infer
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    // output to vector
    std::vector<std::vector<MxBase::ClassInfo>> batchClsInfos = {};
    ret = PostProcess(outputs, batchClsInfos);
    if (ret != APP_ERR_OK)
    {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    // get name
    int index = imgPath.find_last_of("/");
    std::string name = imgPath.substr(index + 1, 23);
    struct stat buffer;
    if (stat(outputPath.c_str(), &buffer) != 0)
    {
        LogError << "Delete file failed.";
    }
    // save result
    ret = SaveResult(outputPath, batchClsInfos, name);
    if (ret != APP_ERR_OK)
    {
        LogError << "save result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
