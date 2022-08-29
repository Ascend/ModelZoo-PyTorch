/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <typeinfo>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <queue>
#include <utility>
#include <fstream>
#include <map>
#include <iostream>
#include "R2U_Net_for_PyTorch.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"



using namespace MxBase;

constexpr auto model_name = "R2U_Net";

APP_ERROR R2U_Net::Init(const InitParam &initParam) {
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

APP_ERROR R2U_Net::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR R2U_Net::ReadImage(const std::string &imgPath, cv::Mat &imageMat)
{
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
    return APP_ERR_OK;
}

APP_ERROR R2U_Net::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat)
{
    static constexpr uint32_t resizeHeight = 224;
    static constexpr uint32_t resizeWidth = 224;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR R2U_Net::Normalize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat)
{
   constexpr size_t ALPHA_AND_BETA_SIZE = 3;
   cv::Mat float32Mat;
   srcImageMat.convertTo(float32Mat, CV_32FC3);

   std::vector<cv::Mat> tmp;
   cv::split(float32Mat, tmp);

   const std::vector<double> mean = {127.5, 127.5, 127.5};
   const std::vector<double> std = {127.5, 127.5, 127.5};
   for (size_t i = 0; i < ALPHA_AND_BETA_SIZE; ++i) {
       tmp[i].convertTo(tmp[i], CV_32FC1, 1 / std[i], - mean[i] / std[i]);
   }
   cv::merge(tmp, dstImageMat);
   return APP_ERR_OK;
}

APP_ERROR R2U_Net::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase)
{
    std::vector<uint32_t> shape_test_image = {1,3, 224, 224};
    uint32_t dataSize_test_image = 1;

    for (uint32_t i = 0; i < shape_test_image.size(); i++) {
            dataSize_test_image *= shape_test_image[i];
    }

    std::vector<cv::Mat> tmp;
    cv::split(imageMat, tmp);
    float *metaData_test_image = new float[dataSize_test_image];
    uint32_t idx = 0;

    for (int i =0; i<3; i++)
    {
        cv::Mat dst = tmp[i].reshape(0,1);
        for (int j =0; j<50176; j++)
        {
            metaData_test_image[idx++] = dst.at<float>(0,j);

        }
    }

    MemoryData memoryDataDst(dataSize_test_image*4, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(metaData_test_image, dataSize_test_image*4, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    tensorBase = TensorBase(memoryDataDst, false, shape_test_image, TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR R2U_Net::Inference(const std::vector<MxBase::TensorBase> &inputs,
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

    if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
    return APP_ERR_OK;
}


APP_ERROR R2U_Net::SaveInferResult(const std::vector<MxBase::TensorBase> &outputs) {
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
    std::string resFileName = "./test_result.txt";
    std::ofstream outfile(resFileName, std::ios::app);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    std::string resultStr;
    for (uint32_t i = 0; i < size; i++) {
        float v = *(value + i);
        resultStr += std::to_string(v) + " ";
    }
    outfile << resultStr << "\n";
    outfile.close();
    return APP_ERR_OK;
}


APP_ERROR R2U_Net::Process(const std::string &imgPath)
{
    cv::Mat srcImageMat;

    APP_ERROR ret = ReadImage(imgPath, srcImageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    ResizeImage(srcImageMat, srcImageMat);

    Normalize(srcImageMat, srcImageMat);

    TensorBase tensorBase;
    ret = CVMatToTensorBase(srcImageMat, tensorBase);
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

    ret = SaveInferResult(outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;

}
