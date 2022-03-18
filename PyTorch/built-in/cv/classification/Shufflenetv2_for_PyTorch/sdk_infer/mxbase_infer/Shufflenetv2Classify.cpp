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
#include <unistd.h>
#include <sys/stat.h>
#include "Shufflenetv2Classify.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using namespace MxBase;
namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
}

APP_ERROR Shufflenetv2Classify::Init(const InitParam &initParam)
{
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
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
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
    if (ret != APP_ERR_OK) {
        LogError << "Resnet50PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    tfile.open("mx_pred_result.txt");
    if (!tfile) {
        LogError << "Open result file failed.";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    return APP_ERR_OK;
}

APP_ERROR Shufflenetv2Classify::DeInit()
{
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    tfile.close();
    return APP_ERR_OK;
}

APP_ERROR Shufflenetv2Classify::ReadImage(const std::string &imgPath, cv::Mat &imageMat)
{
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);

    return APP_ERR_OK;
}

APP_ERROR Shufflenetv2Classify::Resize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat)
{
    static constexpr uint32_t resizeHeight = 352;
    static constexpr uint32_t resizeWidth = 352;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeHeight, resizeWidth));
    return APP_ERR_OK;
}

APP_ERROR Shufflenetv2Classify::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase)
{
    const uint32_t dataSize = imageMat.cols * imageMat.rows * imageMat.channels();
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(imageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {static_cast<uint32_t>(imageMat.rows),
                                   static_cast<uint32_t>(imageMat.cols), static_cast<uint32_t>(imageMat.channels())};
    tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR Shufflenetv2Classify::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                             std::vector<MxBase::TensorBase> &outputs)
{
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
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

APP_ERROR Shufflenetv2Classify::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                               std::vector<std::vector<MxBase::ClassInfo>> &clsInfos)
{
    APP_ERROR ret = post_->Process(inputs, clsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Shufflenetv2Classify::Process(const std::string &imgPath)
{
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    ret = Resize(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }
    TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<std::vector<MxBase::ClassInfo>> BatchClsInfos = {};
    ret = PostProcess(outputs, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    ret = Export(imgPath, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Export result to file failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Shufflenetv2Classify::Export(const std::string &imgPath,
                                       std::vector<std::vector<MxBase::ClassInfo>> &BatchClsInfos)
{
    uint32_t batchIndex = 0;
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    for (const auto &clsInfos: BatchClsInfos) {
        std::string resultStr;
        for (const auto &clsInfo: clsInfos) {
            resultStr += std::to_string(clsInfo.classId) + ",";
        }
        tfile << fileName.substr(0, dot) << " " << resultStr << std::endl;
        if (tfile.fail()) {
            LogError << "Failed to write the result to file.";
            return APP_ERR_COMM_WRITE_FAIL;
        }
        batchIndex++;
    }
    return APP_ERR_OK;
}
