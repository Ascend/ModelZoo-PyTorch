/*
 * Copyright 2022 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-3.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */

#include <iostream>
#include <experimental/filesystem>
#include <vector>
#include "Shufflenetv1Classify.h"
#include "MxBase/Log/Log.h"

namespace fs = std::experimental::filesystem;
namespace {
const uint32_t CLASS_NUM = 1000;
}
std::vector<double> g_inferCost;

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './build/shufflenetv1 ./image_dir'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../data/config/imagenet1000_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/ShuffleNetV1.om";
    auto shufflenetv1 = std::make_shared<Shufflenetv1Classify>();
    APP_ERROR ret = shufflenetv1->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Shufflenetv1Classify init failed, ret=" << ret << ".";
        return ret;
    }

    std::string binDir = argv[1];
    for (auto & entry : fs::directory_iterator(binDir)) {
        LogInfo << "read image path " << entry.path();
        ret = shufflenetv1->Process(entry.path());
        if (ret != APP_ERR_OK) {
            LogError << "Shufflenetv1Classify process failed, ret=" << ret << ".";
            shufflenetv1->DeInit();
            return ret;
        }
    }
    shufflenetv1->DeInit();
    double costSum = 0;
    for (unsigned int i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}