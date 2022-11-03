/*
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

#include "deeppose.h"
#include "MxBase/Log/Log.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

void InitDeepPoseParam(InitParam &initParam) {
    initParam.deviceId = 0;
    initParam.modelPath = "../convert/deeppose.om";
    initParam.csvPath = "./info.csv";
    initParam.imagePath = "../../../coco2017/val2017/";
}

void ReadAnnot(int idx, ImageAnnot &imageAnnot, const InitParam &initParam) {
    ifstream inFile(initParam.csvPath, std::ios::in);
    if (!inFile) {
        LogError << "OPEN csvFile ERROR, csvPath is " << initParam.csvPath << ".";
        exit(1);
    }

    string line;
    string field;
    int lineNum = idx;
    int num = 0;
    while (getline(inFile, line)) {
        num++;
        if(num == lineNum) {
            istringstream sin(line);
            getline(sin, field, ',');
            imageAnnot.imageName = field; 
            getline(sin, field, ','); 
            imageAnnot.center[0] = strtod(field.c_str(), NULL);
            getline(sin, field, ',');
            imageAnnot.center[1] = strtod(field.c_str(), NULL);
            getline(sin, field, ',');
            imageAnnot.scale[0] = strtod(field.c_str(), NULL);
            getline(sin, field, ',');
            imageAnnot.scale[1] = strtod(field.c_str(), NULL);
            getline(sin, field);
            int bboxId = strtod(field.c_str(), NULL);
            imageAnnot.bboxID = to_string(bboxId);
            break;
        }

    }        
    inFile.close();    
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input an integer image index whitch is between 1 and 2985.";
        return APP_ERR_OK;
    }

    int num = atoi(argv[1]); 
    if (num <= 0 || num > 104125) {
        LogError << "image number is invalid, number must be an integer number between 1 and 2985.";
        return APP_ERR_OK;      
    }

    InitParam initParam;
    InitDeepPoseParam(initParam);
    ImageAnnot imageAnnot;
    
    LogInfo << "model path: " << initParam.modelPath;
    LogInfo << "csvfile path: " << initParam.csvPath;

    auto deeppose = std::make_shared<DeepPose>();
    APP_ERROR ret = deeppose->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "DeepPose init failed, ret=" << ret << ".";
        return ret;
    }

    for (int i = 1; i <= num; i++) {
        ReadAnnot(i, imageAnnot, initParam);
        LogInfo << "image name: " << imageAnnot.imageName;

        ret = deeppose->Process(imageAnnot, initParam.imagePath);
        if (ret != APP_ERR_OK) {
            LogError << "DeepPose process failed, ret=" << ret << ".";
            deeppose->DeInit();
            return ret;
        }
    }
    deeppose->DeInit();
    return APP_ERR_OK;
}