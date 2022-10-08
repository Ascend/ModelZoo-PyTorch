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

#include <experimental/filesystem>
#include "MxBase/Log/Log.h"
#include "resnet34.h"
#include <map>
#include <fstream>

namespace fs = std::experimental::filesystem;
std::vector<double> g_inferCost;

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path and ouput path, such as ../utils/preprocess ../preprocess_result.txt";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../data/model/resnet34.om";
    auto resnet34 = std::make_shared<resnet34Infer>();
    APP_ERROR ret = resnet34->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "resnet34Infer init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgDir = argv[1];
    std::string outputDir = argv[2];
    int index = 0;
    for (auto &entry : fs::directory_iterator(imgDir)) {
        index++;
        LogInfo << "read image path " << entry.path();
        ret = resnet34->Process(entry.path(), outputDir);
        if (ret != APP_ERR_OK) {
            LogError << "resnet34Infer process failed, ret=" << ret << ".";
            resnet34->DeInit();
            return ret;
        }
    }
    resnet34->DeInit();
    double costSum = 0;
    for (unsigned int i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size()
            << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum
            << " images/sec.";
    return APP_ERR_OK;
}
