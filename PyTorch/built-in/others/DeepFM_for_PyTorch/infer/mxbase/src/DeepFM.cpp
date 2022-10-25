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
#include "DeepFM.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

std::vector<std::string> split(const std::string& s, const std::string& seperator) {
    std::vector<std::string> result;
    typedef std::string::size_type string_size;
    string_size i = 0;

    while (i != s.size()) {
        uint32_t flag = 0;
        while (i != s.size() && flag == 0) {
            flag = 1;
            for (string_size x = 0; x < seperator.size(); ++x)
                if (s[i] == seperator[x]) {
                    ++i;
                    flag = 0;
                    break;
                }
        }

        flag = 0;
        string_size j = i;
        while (j != s.size() && flag == 0) {
            for (string_size x = 0; x < seperator.size(); ++x)
                if (s[j] == seperator[x]) {
                    flag = 1;
                    break;
                }
            if (flag == 0)
                ++j;
        }
        if (i != j) {
            result.push_back(s.substr(i, j - i));
            i = j;
        }
    }
    return result;
}

APP_ERROR DEEPFM::Init(const InitParam& initParam) {
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

    return APP_ERR_OK;
}

APP_ERROR DEEPFM::DeInit() {
    this->model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR DEEPFM::Inference(const std::vector<MxBase::TensorBase>& inputs, std::vector<MxBase::TensorBase>* outputs) {
    auto dtypes = this->model_->GetOutputDataType();
    for (size_t i = 0; i < this->modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        LogInfo << "----------------------------------shape:";
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)this->modelDesc_.outputTensors[i].tensorDims[j]);
            LogInfo << (uint32_t)this->modelDesc_.outputTensors[i].tensorDims[j] << std::endl;
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

APP_ERROR DEEPFM::PushInputTensor(const std::vector<float>* data, uint32_t index, std::vector<MxBase::TensorBase>* inputs) {
    APP_ERROR ret;
    LogInfo << "--------------------index " << index << " data:";
    const uint32_t dataSize = this->modelDesc_.inputTensors[index].tensorSize;
    float* temp_data = new float[data->size()];
    for (uint32_t i = 0; i < data->size(); i++) {
        *(temp_data + i) = (*data)[i];
        LogInfo << (*data)[i]<<std::endl;
    }
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(temp_data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }
    std::vector<uint32_t> shape;
    if (index == 0) {
        shape.push_back(1);
        shape.push_back(26);
    }
    else {
        shape.push_back(1);
        shape.push_back(1);
        shape.push_back(13);
    }

    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32));
    return APP_ERR_OK;
}

APP_ERROR DEEPFM::ReadInputTensor(const std::string& inputdata, std::vector<MxBase::TensorBase>* inputs) {
    APP_ERROR ret;
    std::vector<std::string> temp_inputdata = split(inputdata, "\t");
    temp_inputdata.erase(temp_inputdata.begin());
    std::vector<float> inputdata_13;
    std::vector<float> inputdata_26;
    std::vector<std::string>::iterator it;
    for (it = temp_inputdata.begin(); it < temp_inputdata.end();it++) {
        if (it - temp_inputdata.begin() < 13) {
            std::istringstream iss(*it);
            float temp_data;
            iss >> temp_data;
            inputdata_13.push_back(temp_data);
        }
        else {
            std::istringstream iss(*it);
            float temp_data;
            iss >> temp_data;
            inputdata_26.push_back(temp_data);
        }
    }
    ret = PushInputTensor(&inputdata_26, 0, inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input_13 ids failed, ret=" << ret << ".";
        return ret;
    }

    ret = PushInputTensor(&inputdata_13, 1, inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input_26 ids failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR DEEPFM::WriteResult(uint32_t index, std::vector<MxBase::TensorBase> outputs, std::string output_data_path){
    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }
    auto dataptr = (float*)outputs[0].GetBuffer();
    std::string fileName = output_data_path + "/" + std::to_string(index) + ".txt";
    LogInfo << fileName;
    std::ofstream tfile(fileName);
    if (tfile.fail()) {
        LogError << "Failed to open result file";
        return APP_ERR_COMM_FAILURE;
    }
    for (uint32_t i = 0; i < outputs.size(); i++) {
        tfile << *(dataptr + i) << std::endl;
    }
    tfile.close();
}

APP_ERROR DEEPFM::Process(const std::string& inputdata, uint32_t index, std::string output_data_path) {
    APP_ERROR ret;
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    ret = ReadInputTensor(inputdata, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Inputdata is invaild.";
        return ret;
    }

    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = WriteResult(index, outputs, output_data_path);
    if (ret != APP_ERR_OK) {
        LogError << "Write result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}