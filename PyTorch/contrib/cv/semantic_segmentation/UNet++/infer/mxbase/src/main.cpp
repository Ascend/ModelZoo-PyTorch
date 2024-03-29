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

#include "Nested_UNet_for_PyTorch.h"
#include "MxBase/Log/Log.h"

namespace {
const uint32_t CLASS_NUM = 2;
const uint32_t MODEL_TYPE = 1;
}  // namespace

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './unet_seg_opencv test.png'";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.modelType = MODEL_TYPE;
    initParam.modelPath = "../convert/output/NestedUNet_npu8p.om";  // Modify according to the actual situation
    NestedUnet unetSeg;
    APP_ERROR ret = unetSeg.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "NestedUnet init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    ret = unetSeg.Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "UnetSegmentation process failed, ret=" << ret << ".";
        unetSeg.DeInit();
        return ret;
    }

    unetSeg.DeInit();
    return APP_ERR_OK;
}