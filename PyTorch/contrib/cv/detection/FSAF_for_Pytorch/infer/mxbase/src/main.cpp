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

#include "fsaf.h"

#include "MxBase/Log/Log.h"

int main() {
    // config the parameters for fsaf model
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../../data/model/fsaf.om";

    // declare and initialize the fsaf model
    FSAF fsaf;
    APP_ERROR ret = fsaf.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FSAF init failed, ret=" << ret << ".";
        return ret;
    }
   
    // coco2017 validation set for object detection
    std::string imgPath = "../../data/input/val2017";

    // directories for saving result
    std::string outputPath = "../output/";
    std::string resultPath = outputPath + "infer_result";
    std::string showPath = outputPath + "show_result";

    // call the process of fsaf model
    ret = fsaf.Process(imgPath, resultPath, showPath);
    if (ret != APP_ERR_OK) {
        LogError << "FSAF process failed, ret=" << ret << ".";
        fsaf.DeInit();
        return ret;
    }

    fsaf.DeInit();
    return APP_ERR_OK;
}
