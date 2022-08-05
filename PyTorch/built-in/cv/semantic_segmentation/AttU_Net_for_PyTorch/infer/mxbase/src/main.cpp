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
#include "AttU_Net_for_PyTorch.h"
#include "MxBase/Log/Log.h"
#include <fstream>

APP_ERROR ReadImagefromTxt(const std::string &txtPath, std::vector<std::vector<float>> &test_image,
                                              std::vector<std::vector<float>> &test_GT_image) {
    auto test_images_path = txtPath + "valid_images.txt";
    auto test_GT_images_path = txtPath + "valid_GT_images.txt";

    std::ifstream test_image_txt(test_images_path);
    std::ifstream test_GT_image_txt(test_GT_images_path);

    std::string ti;
    std::string tgti;
    std::vector<float> tmp_v = {};
    int i = 0;
    while(!test_image_txt.eof() && std::getline(test_image_txt,ti))
    {
      std::string num;
      std::istringstream readstr(ti);
      while(std::getline(readstr, num, ' ')){
        tmp_v.push_back(atof(num.c_str()));
      }
      test_image.push_back(tmp_v);
      tmp_v.clear();
      i++;
    }
    i = 0;
    while(test_GT_image_txt.eof() && std::getline(test_GT_image_txt,tgti))
    {
      std::string num;
      std::istringstream readstr(tgti);
      while(std::getline(readstr, num, ' ')){
        tmp_v.push_back(atof(num.c_str()));
      }
      test_GT_image.push_back(tmp_v);
      tmp_v.clear();
      i++;
    }
    return APP_ERR_OK;
}

void AttU_Net_Param(InitParam *initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../data/models/AttU_Net.om";
}

int main(int argc, char* argv[]) {
    InitParam initParam;
    AttU_Net_Param(&initParam);
    auto attu_net = std::make_shared<AttU_Net>();
    APP_ERROR ret = attu_net->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "AttU_Net init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::vector<float>> test_images = {};
    std::vector<std::vector<float>> test_GT_images = {};
    std::string image_path = "../data/input/";
    ret = ReadImagefromTxt(image_path, test_images, test_GT_images);
    if (ret != APP_ERR_OK) {
        LogError << "AttU_Net read image failed, ret=" << ret << ".";
        attu_net->DeInit();
        return ret;
    }
    for (uint32_t i = 0; i < test_images.size(); i++) {
        ret = attu_net->Process(test_images[i]);
        if (ret != APP_ERR_OK) {
            LogError << "AttU_Net process failed, ret=" << ret << ".";
            attu_net->DeInit();
            return ret;
        }
    }
    attu_net->DeInit();
    LogInfo << "save infer results success";
    return APP_ERR_OK;
}
