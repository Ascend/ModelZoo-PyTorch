/*
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MxBase/Log/Log.h"
#include "fcos.h"

int main() {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../../data/model/fcos.om";
    FCOS fcos;
    APP_ERROR ret = fcos.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FCOS init failed, ret=" << ret << ".";
        return ret;
    }
   
    std::string imgPath = "../../data/input/COCO2017/val2017";
    std::string resultPath = "../data/infer_result";
    std::string showPath = "../data/show_result";
    float PROB_THRES = 0.05;

    ret = fcos.Process(imgPath,resultPath,showPath,PROB_THRES);
    if (ret != APP_ERR_OK) {
        LogError << "FCOS process failed, ret=" << ret << ".";
        fcos.DeInit();
        return ret;
    }

    fcos.DeInit();
    return APP_ERR_OK;
}
