/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "Nested_UNet_for_PyTorch.h"

#include <memory>
#include <vector>
#include <string>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"


APP_ERROR NestedUnet::Init(const InitParam &initParam) {
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

APP_ERROR NestedUnet::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR NestedUnet::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR NestedUnet::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat,
                                   MxBase::ResizedImageInfo &resizedImageInfo) {
    static constexpr uint32_t resizeHeight = 96;
    static constexpr uint32_t resizeWidth = 96;

    resizedImageInfo.heightOriginal = srcImageMat.rows;
    resizedImageInfo.heightResize = resizeHeight;
    resizedImageInfo.widthOriginal = srcImageMat.cols;
    resizedImageInfo.widthResize = resizeWidth;
    resizedImageInfo.resizeType = MxBase::RESIZER_STRETCHING;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeHeight, resizeWidth));
    return APP_ERR_OK;
}

APP_ERROR NestedUnet::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR NestedUnet::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR NestedUnet::PostProcess(std::vector<MxBase::TensorBase> &inputs,
                                        const MxBase::ResizedImageInfo &resizedInfo, cv::Mat &output) {
    MxBase::TensorBase &tensor = inputs[0];
    int ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }

    uint32_t imgHeight = resizedInfo.heightOriginal;
    uint32_t imgWidth = resizedInfo.widthOriginal;

    uint32_t outputModelWidth = tensor.GetShape()[MxBase::VECTOR_THIRD_INDEX];
    uint32_t outputModelHeight = tensor.GetShape()[MxBase::VECTOR_FOURTH_INDEX];

    cv::Mat imageMat(outputModelHeight, outputModelWidth, CV_32FC1);
    auto data = reinterpret_cast<float(*)[outputModelWidth][outputModelHeight]>(tensor.GetBuffer());
    for (size_t x = 0; x < outputModelHeight; ++x) {
        for (size_t y = 0; y < outputModelWidth; ++y) {
            imageMat.at<float>(x, y) = sigmoid(data[0][x][y]);
        }
    }
    cv::resize(imageMat, imageMat, cv::Size(imgWidth, imgHeight), cv::INTER_CUBIC);
    cv::Mat argmax(imgHeight, imgWidth, CV_8UC1);
    for (size_t x = 0; x < imgHeight; ++x) {
        for (size_t y = 0; y < imgWidth; ++y) {
            argmax.at<uchar>(x, y) = imageMat.at<float>(x, y) * 255;
        }
    }
    output = argmax;
    return APP_ERR_OK;
}

APP_ERROR NestedUnet::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    MxBase::ResizedImageInfo resizedImageInfo;
    ret = ResizeImage(imageMat, imageMat, resizedImageInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ResizeImage failed, ret=" << ret << ".";
        return ret;
    }

    MxBase::TensorBase tensorBase;
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

    cv::Mat output;
    ret = PostProcess(outputs, resizedImageInfo, output);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    std::string resultPath = "./infer_result/";
    size_t pos = imgPath.find_last_of("/");
    resultPath = resultPath + imgPath.substr(pos+1);
    cv::imwrite(resultPath, output);
    return APP_ERR_OK;
}

float NestedUnet::sigmoid(float x) {
    return 1.f / (1.f + exp(-x));
}