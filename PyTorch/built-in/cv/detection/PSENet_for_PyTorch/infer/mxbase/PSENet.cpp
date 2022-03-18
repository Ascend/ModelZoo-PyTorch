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

#include "PSENet.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using namespace MxBase;
namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
}

APP_ERROR PSENet::Init(const InitParam &initParam)
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
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("KERNEL_NUM", std::to_string(initParam.kernelNum));
    configData.SetJsonValue("PSE_SCALE", std::to_string(initParam.pseScale));
    configData.SetJsonValue("MIN_KERNEL_AREA", std::to_string(initParam.minKernelArea));
    configData.SetJsonValue("MIN_SCORE", std::to_string(initParam.minScore));
    configData.SetJsonValue("MIN_AREA", std::to_string(initParam.minArea));


    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::PSENetPostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "PSENetPostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR PSENet::DeInit()
{
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR PSENet::ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor)
{
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, output);
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, MemoryData::MemoryType::MEMORY_DVPP, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::vector<uint32_t> shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    tensor = TensorBase(memoryData, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR PSENet::Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor)
{
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};

    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    const uint32_t resizeHeight = 704;
    const uint32_t resizeWidth = 1216;
    MxBase::ResizeConfig resize = {};
    resize.height = resizeHeight;
    resize.width = resizeWidth;
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, MemoryData::MemoryType::MEMORY_DVPP, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MemoryHelper::MxbsFree(memoryData);
        MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    outputTensor = TensorBase(memoryData, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR PSENet::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR PSENet::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                      std::vector<std::vector<MxBase::TextObjectInfo>> &txtObjInfos,
                      const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos
                     )
{
    LogInfo << "real start to psenet post process";
    APP_ERROR ret = post_->Process(inputs, txtObjInfos, resizedImageInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "real end to psenet post process";
    return APP_ERR_OK;
}


APP_ERROR PSENet::SaveResult(const std::string &imgPath, const std::vector<std::vector<MxBase::TextObjectInfo>> &batchTextObjectInfos)
{
    LogInfo << "image path" << imgPath;
    std::string file_name = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = file_name.find_last_of(".");
    std::string resFileName = "psenet_result/" + fileName.substr(0, dot) + ".txt";
    LogInfo << "file path for saving result" << resFileName;

    std::ofstream outfile(resFileName);
    if (outfile.fail()) {
        LogError << "Failed to open result file";
        return APP_ERR_COMM_FAILURE;
    }

    uint32_t batchIndex = 0;
    for (auto textObjectInfos : batchTextObjectInfos) {
        for (auto textObjectInfo : textObjectInfos) {
            std::string resultStr = "";
            resultStr += std::to_string(textObjectInfo.x3) + "," + std::to_string(textObjectInfo.y3) + "," +
            std::to_string(textObjectInfo.x0) + "," + std::to_string(textObjectInfo.y0) + "," +
            std::to_string(textObjectInfo.x1) + "," + std::to_string(textObjectInfo.y1) + "," +
            std::to_string(textObjectInfo.x2) + "," + std::to_string(textObjectInfo.y2);
            outfile << resultStr << std::endl;
        }
        batchIndex++;
    }
    outfile.close();
    return APP_ERR_OK;
}


APP_ERROR PSENet::Process(const std::string &imgPath)
{
    TensorBase image;
    LogInfo << "input image path" << imgPath;
    APP_ERROR ret = ReadImage(imgPath, image);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "read input image success";

    TensorBase resizeImage;
    LogInfo << "start to resize image";
    ret = Resize(image, resizeImage);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }
    LogError << "input file size after dvpp, size=" << resizeImage.GetShape()[0] << " " << resizeImage.GetShape()[1] << ".";
    LogInfo << "resize image success";

    LogInfo << "start to push resize image";
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(resizeImage);
    LogInfo << "end to push resize image";

    LogInfo << "start inference";
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "inference success";

    LogInfo << "start to push resized image info";
    std::vector<std::vector<TextObjectInfo>> batchTextObjectInfos = {};
    std::vector<MxBase::ResizedImageInfo> resizedImageInfos = {};
    ResizedImageInfo resizedImageInfo;
    resizedImageInfo.widthResize = int(1216);
    resizedImageInfo.heightResize = int(704);
    resizedImageInfo.widthOriginal = int(1280);
    resizedImageInfo.heightOriginal = int(740);
    resizedImageInfos.push_back(resizedImageInfo);
    LogInfo << "end to push resized image info";

    LogInfo << "start to post psenet process";
    ret = PostProcess(outputs, batchTextObjectInfos, resizedImageInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "end to post psenet process";

    ret = SaveResult(imgPath, batchTextObjectInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}
