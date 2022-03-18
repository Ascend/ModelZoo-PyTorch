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

#include "PSENet.h"
#include "MxBase/Log/Log.h"
#include <string>
#include <iostream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './psenet image_dir'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 2;
    initParam.kernelNum = 7;
    initParam.pseScale = 1.0;
    initParam.minKernelArea = 5.0;
    initParam.minScore = 0.93;
    initParam.minArea = 800.0;
    initParam.labelPath = "../models/imagenet1000_clsidx_to_labels_empty.names";

    initParam.checkTensor = true;
    initParam.modelPath = "../models/psenet.om";
    auto psenet = std::make_shared<PSENet>();
    APP_ERROR ret = psenet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "psenet init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgDir = argv[1];
    for (auto & entry : fs::directory_iterator(imgDir)) {
        LogInfo << "read image path" << entry.path();

        ret = psenet->Process(entry.path());
        if (ret != APP_ERR_OK) {
            LogError << "psenet process failed, ret=" << ret << ".";
            psenet->DeInit();
            return ret;
        }
    }
    psenet->DeInit();
    return APP_ERR_OK;
}
