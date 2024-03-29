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
#include "UNet.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"
#include <typeinfo>

//Init
APP_ERROR UNet::Init(const InitParam &initParam) {
    this->deviceId_ = initParam.deviceId;
    this->outputDataPath_ = initParam.outputDataPath;
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

    this->model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = this->model_->Init(initParam.modelPath, this->modelDesc_);
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

    return APP_ERR_OK;
}

APP_ERROR UNet::DeInit() {
    this->model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR UNet::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * 3 * 4;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);

    static unsigned char data[3 * 96 * 96 * 4];
    //modify
    uint32_t W=(uint32_t)imageMat.cols,H=(uint32_t)imageMat.rows;
    for (size_t h = 0; h < H; h++)
        for (size_t w = 0; w < W; w++)
            for (size_t c = 0; c < 3; c++) {
                data[(c * W * H + h * W + w)*4 + 0] = imageMat.data[(h * W * 3 + w * 3 + c)*4 + 0];
                data[(c * W * H + h * W + w)*4 + 1] = imageMat.data[(h * W * 3 + w * 3 + c)*4 + 1];
                data[(c * W * H + h * W + w)*4 + 2] = imageMat.data[(h * W * 3 + w * 3 + c)*4 + 2];
                data[(c * W * H + h * W + w)*4 + 3] = imageMat.data[(h * W * 3 + w * 3 + c)*4 + 3];
            }
    MxBase::MemoryData memoryDataSrc(data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, 3, static_cast<uint32_t>(imageMat.rows), static_cast<uint32_t>(imageMat.cols)};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

//Inference
APP_ERROR UNet::Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = this->model_->GetOutputDataType();
    for (size_t i = 0; i < this->modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)this->modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, this->deviceId_);
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
    APP_ERROR ret = this->model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);

    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR UNet::BinWriteResult(const std::string &imageFile, std::vector<MxBase::TensorBase> outputs) {
    for (size_t i = 0; i < 1; ++i) {  //  when model's aux==True, we don't use the other two outputs.
        APP_ERROR ret = outputs[i].ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "tohost fail.";
            return ret;
        }
        void *netOutput = outputs[i].GetBuffer();
        std::vector<uint32_t> out_shape = outputs[i].GetShape();
        int pos = imageFile.rfind('/');
        std::string fileName(imageFile, pos + 1);
        fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), "_0.bin");
        std::string outFileName = this->outputBinPath_ + "/" + fileName;
        FILE *outputFile_ = fopen(outFileName.c_str(), "wb");
        fwrite(netOutput, out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3], sizeof(float), outputFile_);
        fclose(outputFile_);
    }
    return APP_ERR_OK;
}

//save mask
APP_ERROR UNet::ImageWriteResult(MxBase::TensorBase *tensor,cv::Mat &output) {
    APP_ERROR ret = tensor->ToHost();
    if (ret != APP_ERR_OK) {
        LogError << "ToHost failed.";
        return ret;
    }
    // 1 x 1 x 96 x 96
    auto data = reinterpret_cast<float *>(tensor->GetBuffer());
    float inferPixel;
    for (size_t x = 0; x < 96; ++x) {
        for (size_t y = 0; y < 96; ++y) {
            inferPixel = *(data + x * 96 + y);
            inferPixel = (1 / (1 + exp(-inferPixel)))*255;
            uchar pixel = uchar(inferPixel);
            output.at<uchar>(x, y) = pixel;
        }
    }
    return APP_ERR_OK;
}

//Process
APP_ERROR UNet::Process(const std::string &inferPath, const std::string &fileName) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputIdsFile = inferPath + fileName;
    // start_nomalization
    cv::Mat src = cv::imread(inputIdsFile, cv::IMREAD_COLOR);
    //cv::cvtColor(src, src, cv::COLOR_RGB2BGR);
    src.convertTo(src, CV_32F);
    //src = src / 255.0;
    std::vector<float> mean_value{123.675, 116.28, 103.53};
    std::vector<float> std_value{58.395, 57.12, 57.375};
    cv::Mat dst;
    std::vector<cv::Mat> bgrChannels;
    cv::split(src, bgrChannels);
    for (auto i = 0; i < bgrChannels.size(); i++)
    {
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std_value[i], (0.0 - mean_value[i]) / std_value[i]);
    }
    cv::merge(bgrChannels, dst);
    dst.convertTo(dst, CV_32F);
    dst = dst / 255.0;
    //end
    //to tenserbase and insert to inputs
    //
    MxBase::TensorBase tensorBase;
    APP_ERROR ret = CVMatToTensorBase(dst, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "Read input ids failed, ret=" << ret << ".";
        return ret;
    }
    //to tenserbase
    //create tenserbase for save outputs and satrt to inference
    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    //end
    //save result of inference (bin_type)
    ret = BinWriteResult(fileName, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    //save to the file_path that had built

    int outputModelHeight = outputs[0].GetShape()[MxBase::VECTOR_THIRD_INDEX];
    int outputModelWidth = outputs[0].GetShape()[MxBase::VECTOR_FOURTH_INDEX];
    cv::Mat output(outputModelHeight, outputModelWidth, CV_8UC1); //8U指的可能是Unsigned int8
    ret = ImageWriteResult(&outputs[0], output);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    std::string outFileName = this->outputDataPath_ + "/" + fileName;
    size_t pos = outFileName.find_last_of(".");
    outFileName.replace(outFileName.begin() + pos, outFileName.end(), ".png");
    cv::imwrite(outFileName, output);

    return APP_ERR_OK;
}
