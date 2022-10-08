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

#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "ICNet.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitProtonetParam(InitParam *initParam, const std::string &model_path, const std::string &output_data_path, const std::string &output_bin_path) {
    initParam->deviceId = 0;
    initParam->modelPath = model_path;
    initParam->outputDataPath = output_data_path;
    initParam->outputBinPath = output_bin_path;
}

APP_ERROR ReadFilesFromPath(const std::string &path, std::vector<std::string> *files) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;
    if ((dir = opendir(path.c_str())) == NULL) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
        }
    while ((ptr = readdir(dir)) != NULL) {
        if (ptr->d_type == 8) {
            files->push_back(ptr->d_name);
        }
    }
    closedir(dir);
    return APP_ERR_OK;
}

int main(int argc, char *argv[]) {
    LogInfo << "=======================================  !!!Parameters setting!!! "
            << "========================================";
    std::string model_path = argv[1];
    LogInfo << "==========  loading model weights from: " << model_path;

    std::string input_data_path = argv[2];
    LogInfo << "==========  input data path = " << input_data_path;

    std::string output_data_path = argv[3];
    LogInfo << "==========  output data path = " << output_data_path;

    std::string output_bin_path = argv[4];
    LogInfo << "==========  output bin path = " << output_bin_path;

    LogInfo << "========================================  !!!Parameters setting!!! "
            << "========================================";

    InitParam initParam;
    InitProtonetParam(&initParam, model_path, output_data_path, output_bin_path);
    auto icnet = std::make_shared<ICNet>();
    APP_ERROR ret = icnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "ICNet init failed, ret=" << ret << ".";
        return ret;
    }

    std::string paths[3];
    paths[0]=input_data_path+"/munster/";
    paths[1]=input_data_path+"/frankfurt/";
    paths[2]=input_data_path+"/lindau/";
    for (int i=0;i<3;i++){
        std::vector<std::string> files;
        ret = ReadFilesFromPath(paths[i], &files);
        if (ret != APP_ERR_OK) {
            LogError << "Read files from path failed, ret=" << ret << ".";
            return ret;
        }
        // do infer
        for (uint32_t j = 0; j < files.size(); j++) {//uint32_t
            LogInfo << "Processing: " + std::to_string(j + 1) + "/" + std::to_string(files.size()) + " ---> " + files[j];
            ret = icnet->Process(paths[i], files[j]);
            if (ret != APP_ERR_OK) {
                LogError << "ICNet process failed, ret=" << ret << ".";
                icnet->DeInit();
                return ret;
           }
       }
    }

    LogInfo << "infer succeed and write the result data with binary file !";

    icnet->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " bin/sec.";
    LogInfo << "==========  The infer result has been saved in ---> " << output_data_path;
    return APP_ERR_OK;
}