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
#include "MxBase/Log/Log.h"
#include "Gluon_ResNet50_v1b.h"
#include <map>
#include <fstream>

namespace fs = std::experimental::filesystem;
std::vector<double> g_inferCost;

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> &imgFiles)
{
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        imgFiles.emplace_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path and ouput path, such as '../data/input/preprocess ../data/input/preprocess_result.txt";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../../data/model/gluon_resnet50_v1b_bs1.om";
    initParam.labelPath = "../../imagenet1000_clsidx_to_labels.names";
    auto gluon_resnet50_v1b = std::make_shared<Gluon_ResNet50_v1bInfer>();
    APP_ERROR ret = gluon_resnet50_v1b->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Gluon_ResNet50_v1bInfer init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgDir = argv[1];
    std::string outputDir_preprocess_result = argv[2];

    std::vector<std::string> imgFilePaths;
    ScanImages(imgDir, imgFilePaths);

    int index = 0;
    for (auto &entry : fs::directory_iterator(imgDir)) {
        std::string &imgFile = imgFilePaths[index];
        index++;
        LogInfo << "read image path " << entry.path();
        LogInfo << "imgFile: " << imgFile;
        ret = gluon_resnet50_v1b->Process(imgFile, entry.path(), outputDir_preprocess_result);
        if (ret != APP_ERR_OK) {
            LogError << "Gluon_ResNet50_v1bInfer process failed, ret=" << ret << ".";
            gluon_resnet50_v1b->DeInit();
            return ret;
        }
    }
    gluon_resnet50_v1b->DeInit();
    double costSum = 0;
    for (unsigned int i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size()
            << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum
            << " images/sec.";
    return APP_ERR_OK;
}