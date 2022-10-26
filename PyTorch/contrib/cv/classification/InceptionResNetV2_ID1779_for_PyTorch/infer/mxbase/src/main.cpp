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
#include "InceptionResnetV2.h"
#include "MxBase/Log/Log.h"
#include <dirent.h>
#include <map>
#include <fstream>

namespace fs = std::experimental::filesystem;
std::vector<double> g_inferCost;

namespace
{
    const uint32_t CLASS_NUM = 1001;
} // namespace

int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        LogWarn << "Please input image path and ouput path, such as '../data/input/preprocess ../data/input/preprocess_result.txt";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;

    initParam.labelPath = "../data/imagenet1000_clsidx_to_labels.names";
    initParam.modelPath = "../data/model/InceptionResNetV2_npu_16.om";
    auto inceptionresnet = std::make_shared<InceptionResnetV2Infer>();
    APP_ERROR ret = inceptionresnet->Init(initParam);
    if (ret != APP_ERR_OK)
    {
        LogError << "InceptionResnetV2Infer init failed, ret=" << ret << ".";
        return ret;
    }

    // sorted filepath by the alphabetical order of filename
    std::string imgDir = argv[1];
    std::string outputDir = argv[2];
    std::vector<fs::path> sorted_path;
    for (auto &entry : fs::directory_iterator(imgDir))
        sorted_path.emplace_back(entry.path());
    sort(sorted_path.begin(), sorted_path.end());

    for (auto &filepath : sorted_path)
    {
        LogInfo << "read image path " << filepath;
        ret = inceptionresnet->Process(filepath, outputDir);
        if (ret != APP_ERR_OK)
        {
            LogError << "InceptionResentV2Classify process failed, ret=" << ret << ".";
            inceptionresnet->DeInit();
            return ret;
        }
    }

    inceptionresnet->DeInit();
    double costSum = 0;
    for (unsigned int i = 0; i < g_inferCost.size(); i++)
    {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size()
            << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum
            << " images/sec.";
    return APP_ERR_OK;
}
