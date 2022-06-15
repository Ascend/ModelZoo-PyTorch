/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <dirent.h>
#include <fstream>
#include "MxBase/Log/Log.h"
#include "DnCNN.h"

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> *imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path << path.c_str();
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        imgFiles->emplace_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    InitParam initParam{};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/DnCNN.om";

    auto model_DnCNN = std::make_shared<DnCNN>();
    APP_ERROR ret = model_DnCNN->Init(initParam);
    if (ret != APP_ERR_OK) {
        model_DnCNN->DeInit();
        LogError << "DnCNN init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    std::vector<std::string> imagesPath;
    ret = ScanImages(imgPath, &imagesPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    float avg_psnr = 0;
    for (auto &imgFile: imagesPath) {
        float psnr;
        LogInfo << "Processing " << imgFile;
        ret = model_DnCNN->Process(imgFile, &psnr);
        if (ret != APP_ERR_OK) {
            LogError << "DnCNN process failed, ret=" << ret << ".";
            model_DnCNN->DeInit();
            return ret;
        }
        LogInfo << "psnr value: " << psnr;
        avg_psnr += psnr;
    }
    model_DnCNN->DeInit();

    LogInfo << "final psnr value: " << avg_psnr / imagesPath.size();

    double total_time = model_DnCNN->GetInferCostMilliSec() / 1000;
    LogInfo << "inferance total cost time: " << total_time << ", FPS: "<< imagesPath.size() / total_time;
    return APP_ERR_OK;
}

