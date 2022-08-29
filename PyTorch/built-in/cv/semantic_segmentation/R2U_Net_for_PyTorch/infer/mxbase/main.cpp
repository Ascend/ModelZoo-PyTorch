/*
 * Copyright 2022 Huawei Technologies Co., Ltd
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ou may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include "R2U_Net_for_PyTorch.h"
#include "MxBase/Log/Log.h"
#include <fstream>

void R2U_Net_Param(InitParam *initParam, const std::string &model_path) {
    initParam->deviceId = 0;
    initParam->modelPath = model_path;
}

int main(int argc, char* argv[]) {
    InitParam initParam;
    std::string imgPath = argv[1];
    std::string model_path = argv[2];
    R2U_Net_Param(&initParam, model_path);
    auto attention_r2u_net = std::make_shared<R2U_Net>();
    APP_ERROR ret = attention_r2u_net->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "R2U_Net init failed, ret=" << ret << ".";
        return ret;
    }

    ret = attention_r2u_net->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "R2U_Net process failed, ret=" << ret << ".";
        attention_r2u_net->DeInit();
        return ret;
    }

    attention_r2u_net->DeInit();
    LogInfo << "save infer results success";
    return APP_ERR_OK;
}
