/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "DeeplabV3.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 21;
    const uint32_t MODEL_TYPE = 1;
    const uint32_t FRAMEWORK_TYPE = 1;
}

int main(int argc, char *argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './deeplabv3 test.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.modelType = MODEL_TYPE;
    initParam.labelPath = "../sdk/etc/deeplabv3.names";
    initParam.modelPath = "../deeplabv3plus_rgb.om";
    initParam.checkModel = true;
    initParam.frameworkType = FRAMEWORK_TYPE;

    DeeplabV3 deeplabV3;
    APP_ERROR ret = deeplabV3.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "DeeplabV3 init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    ret = deeplabV3.Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "DeeplabV3 process failed, ret=" << ret << ".";
        deeplabV3.DeInit();
        return ret;
    }
    deeplabV3.DeInit();
    return APP_ERR_OK;
}
