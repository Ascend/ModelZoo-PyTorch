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

#include "EfficientNetB5Classify.h"
#include "MxBase/Log/Log.h"
#include <algorithm>
#include <dirent.h>

namespace {
const uint32_t CLASS_NUM = 1000;

bool IsJpegSuffix(const std::string &imgFile)
{
    std::string::size_type indexOfDot = imgFile.find_last_of('.');
    if (indexOfDot == std::string::npos) {
        return false;
    }
    std::string suffix = imgFile.substr(indexOfDot + 1);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
    return suffix == "jpg" || suffix == "jpeg";
}

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> &imgFiles)
{
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }
        if (!IsJpegSuffix(fileName)) {
            LogWarn << "image file does not end with jpg or jpeg : " << fileName;
            continue;
        }
        imgFiles.emplace_back(path + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './efficientnetb0 /path/to/jpeg_image_dir'.";
        return APP_ERR_OK;
    }

    std::string imgPath = argv[1];
    if (imgPath[imgPath.length() - 1] != '/') {
        imgPath.push_back('/');
    }
    std::vector<std::string> imgFiles;
    APP_ERROR ret = ScanImages(imgPath, imgFiles);
    if (ret != APP_ERR_OK) {
        LogError << "ScanImages failed, ret=" << ret << ".";
        return ret;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../data/imagenet1000_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/efficientnetb5_npu_16_1.om";
    auto efficientnetb0 = std::make_shared<EfficientNetB5Classify>();
    ret = efficientnetb0->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "EfficientNetB0Classify init failed, ret=" << ret << ".";
        return ret;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    for (auto &imgFile: imgFiles) {
        ret = efficientnetb0->Process(imgFile);
        if (ret != APP_ERR_OK) {
            LogError << "EfficientNetB0Classify process failed, ret=" << ret << ".";
            efficientnetb0->DeInit();
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    efficientnetb0->DeInit();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFiles.size() / efficientnetb0->GetInferCostTimeMs();
    LogInfo << "[Total Process Delay] cost: " << costMs << " ms, fps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}
