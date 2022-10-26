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
#include "Gluon_ResNet50_v1b.h"
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
using namespace MxBase;

APP_ERROR Gluon_ResNet50_v1bInfer::Init(const InitParam &initParam) {

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
    const std::string softmax = "false" ;
    const std::string checkTensor = "true";

    configData.SetJsonValue("CLASS_NUM", std::to_string(1000));
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
    return APP_ERR_OK;
}

APP_ERROR Gluon_ResNet50_v1bInfer::BinToTensorBase(const std::string file_path, MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }
    float *img_data = new float[dataSize];
    std::ifstream inF(file_path, std::ios::binary);
    inF.read(reinterpret_cast<char*>(img_data), sizeof(float) * (dataSize));
    inF.close();
    MxBase::MemoryData memoryDataDst(dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(&img_data[0]), dataSize * 4,
                                     MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, 3, 224, 224};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    delete[] img_data;
    return APP_ERR_OK;
}

APP_ERROR Gluon_ResNet50_v1bInfer::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Gluon_ResNet50_v1bInfer::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    dynamicInfo.batchSize = 1;
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

APP_ERROR Gluon_ResNet50_v1bInfer::PostProcessEval(std::vector<MxBase::TensorBase> *outputs,
                                        std::vector<float> *predict) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0] * outputShape[1];
    void *data = tensor.GetBuffer();
    for (uint32_t i = 0; i < length; i++) {
        float value = *(reinterpret_cast<float *>(data) + i);
        predict->push_back(value);
    }
    return APP_ERR_OK;
}

APP_ERROR Gluon_ResNet50_v1bInfer::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                        std::vector<std::vector<MxBase::ClassInfo>> &clsInfos)
{
    APP_ERROR ret = post_->Process(inputs, clsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Gluon_ResNet50_v1bInfer::WriteResult(const std::string &fileName, const std::vector<float> &predict,
                                        const std::string name) {
    // create result file
    std::ofstream tfile(fileName, std::ofstream::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << fileName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // write inference result into file
    LogInfo << "==============================================================";
    LogInfo << "Infer finished!";

    tfile<<name;
    for (uint32_t i = 0; i < predict.size(); i++){
        tfile << " " << predict[i];
    }
    tfile << std::endl;

    LogInfo << "==============================================================";
    tfile.close();
    return APP_ERR_OK;
}

APP_ERROR Gluon_ResNet50_v1bInfer::SaveResult(const std::string &imgFile, const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos)
{

    LogInfo << "image path" << imgFile;
    std::string fileName = imgFile.substr(imgFile.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    std::string resFileName = "result/" + fileName.substr(0, dot) + "_1.txt";
    LogInfo << "file path for saving result" << resFileName;

    std::ofstream outfile(resFileName);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }

    uint32_t batchIndex = 0;
    for (auto clsInfos : batchClsInfos) {
        std::string resultStr;
        for (auto clsInfo : clsInfos) {
            LogDebug << " className:" << clsInfo.className << " confidence:" << clsInfo.confidence <<
            " classIndex:" <<  clsInfo.classId;
            resultStr += std::to_string(clsInfo.classId) + " ";
        }

        outfile << resultStr << std::endl;
        batchIndex++;
    }
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR Gluon_ResNet50_v1bInfer::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat)
{
    static constexpr uint32_t resizeHeight = 224;
    static constexpr uint32_t resizeWidth = 224;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}



APP_ERROR Gluon_ResNet50_v1bInfer::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase)
{
    const uint32_t dataSize =  imageMat.cols *  imageMat.rows * YUV444_RGB_WIDTH_NU;
    LogInfo << "image size after crop" << imageMat.cols << " " << imageMat.rows;
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(imageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imageMat.rows * YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR Gluon_ResNet50_v1bInfer::ReadImage(const std::string &imgPath, cv::Mat &imageMat)
{
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}


APP_ERROR Gluon_ResNet50_v1bInfer::Process(const std::string &imgFile, const std::string &imgPath, const std::string &outputDir_preprocess_result)
{
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgFile, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

//    ResizeImage(imageMat, imageMat);

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    ret = BinToTensorBase(imgPath, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "BinToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

     inputs.push_back(tensorBase);

     auto startTime = std::chrono::high_resolution_clock::now();
     ret = Inference(inputs, outputs);
     auto endTime = std::chrono::high_resolution_clock::now();
     double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
     inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    //postprocess
    std::vector<std::vector<MxBase::ClassInfo>> BatchClsInfos = {};
    ret = PostProcess(outputs, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveResult(imgFile, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    //eval
    std::vector<float> predict;
    ret = PostProcessEval(&outputs, &predict);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcessEval failed, ret=" << ret << ".";
        return ret;
    }
    // get name
    int index = imgPath.find_last_of("/");
    std::string name = imgPath.substr(index + 1, 23);
    // save result
    ret = WriteResult(outputDir_preprocess_result, predict, name);
    if (ret != APP_ERR_OK) {
        LogError << "save result failed, ret=" << ret << ".";
        return ret;
    }


    return APP_ERR_OK;
}