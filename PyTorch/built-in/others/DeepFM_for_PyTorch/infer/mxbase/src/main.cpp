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

#include <iostream>
#include <fstream>
#include <vector>
#include "DeepFM.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitProtonetParam(InitParam* initParam, const std::string& model_path, const std::string& output_data_path) {
    initParam->deviceId = 0;
    initParam->modelPath = model_path;
    initParam->outputDataPath = output_data_path;
}

APP_ERROR ReadFilesFromPath(const std::string& path, std::vector<std::string>* inputdatas) {
    std::ifstream input_file(path.c_str());
    if (!input_file.is_open()) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::string line;
    while (std::getline(input_file, line)) {
        inputdatas->push_back(line);
    }

    inputdatas->erase(inputdatas->begin());
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    //Load Paramenters
    LogInfo << "=======================================  Parameters setting" << \
        "========================================";
    std::string model_path = argv[1];
    LogInfo << "==========  loading model weights from:" << model_path;

    std::string input_data_path = argv[2];
    LogInfo << "==========  input data path = " << input_data_path;

    std::string output_data_path = argv[3];
    LogInfo << "==========  output data path = " << output_data_path << \
        " WARNING: please make sure that this folder is created in advance!!!";

    //Init model
    LogInfo << "========================================  Loading model " << \
        "========================================";
    InitParam initParam;
    InitProtonetParam(&initParam, model_path, output_data_path);
    auto deepfm = std::make_shared<DEEPFM>();
    APP_ERROR ret = deepfm->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "ecapatdnn init failed, ret=" << ret << ".";
        return ret;
    }
    else {
        LogInfo << "Loading model success";
    }

    //Load data 
    LogInfo << "========================================  Loading Data " << \
        "========================================";
    std::vector<std::string> inputdatas;
    ret = ReadFilesFromPath(input_data_path, &inputdatas);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }
    else {
        LogInfo << "Loading Data success";
    }

    //Start infer
    LogInfo << "========================================  Start infer " << \
        "========================================";
    for (uint32_t i = 0; i < inputdatas.size(); i++) {
        LogInfo << "Processing: " + std::to_string(i + 1) + "/" + std::to_string(inputdatas.size()) + " :\n " + inputdatas[i];
        ret = deepfm->Process(inputdatas[i], i, output_data_path);
        if (ret != APP_ERR_OK) {
            LogError << "deepfm process failed, ret=" << ret << ".";
            deepfm->DeInit();
            return ret;
        }
    }

    //Record the result
    LogInfo << "========================================  Record the result " << \
        "========================================";
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer data sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " bin/sec.";
    LogInfo << "==========  The infer result has been saved in ---> " << output_data_path;
    return APP_ERR_OK;
}