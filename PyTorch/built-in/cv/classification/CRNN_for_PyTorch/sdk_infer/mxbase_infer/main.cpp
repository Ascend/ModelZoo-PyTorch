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

#include "CrnnOcr.h"
#include "MxBase/Log/Log.h"
#include <dirent.h>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
namespace {
const uint32_t CLASS_NUM = 37;
}
std::vector<double> g_inferCost;

int main(int argc, char* argv[])
{
    if (argc <= 2) {
        LogWarn << "Please input image dir and limit, such as './resnext50' -1";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "./crnn-label.names";
    initParam.objectNum = 26;
    initParam.blankIndex = 37;
    initParam.withArgMax = false;
    initParam.modelPath = "../models/crnn.om";
    auto crnn = std::make_shared<CrnnOcr>();
    APP_ERROR ret = crnn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "crnn init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgDir = argv[1];
    long limit = std::strtol(argv[2], nullptr, 0);
    long index = 0;
    for (auto & entry : fs::directory_iterator(imgDir)) {
        if (limit > 0 && index == limit) {
            break;
        }
        index++;
        LogInfo << "read image path " << entry.path();
        ret = crnn->Process(entry.path());
        if (ret != APP_ERR_OK) {
            LogError << "crnn process failed, ret=" << ret << ".";
            crnn->DeInit();
            return ret;
        }
    }
    crnn->DeInit();
    double costSum = 0;
    for (unsigned int i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}