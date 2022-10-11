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
#include "InceptionV3.h"
#include <map>
#include <iostream>
#include <vector>
#include <fstream>

namespace fs = std::experimental::filesystem;
namespace {
const uint32_t CLASS_NUM = 1000;
}
std::vector<double> g_inferCost;

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path and ouput path";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.labelPath = "../data/config/imagenet1000_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.classNum = CLASS_NUM;
    initParam.softmax = false;
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../convert/output/inceptionV3.om";
    auto inceptionV3 = std::make_shared<InceptionV3>();
    APP_ERROR ret = inceptionV3->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "InceptionV3 init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgDir = argv[1];
    std::string outputDir = argv[2];
    int index = 0;
    for (auto &entry : fs::directory_iterator(imgDir)) {
        index++;
        LogInfo << "read image path " << entry.path();
        ret = inceptionV3->Process(entry.path());
        if (ret != APP_ERR_OK) {
            LogError << "InceptionV3 process failed, ret=" << ret << ".";
            inceptionV3->DeInit();
            return ret;
        }
    }
    inceptionV3->DeInit();
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
