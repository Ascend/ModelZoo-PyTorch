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

#include <string.h>
#include <math.h>
#include <dirent.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

#include "fcos.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"

using namespace MxBase;

std::vector<std::string> classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

APP_ERROR FCOS::Init(const InitParam& initParam) {
    // Equipment initialization
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    // Context initialization
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    
    // Load model
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR FCOS::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

std::vector<std::string> FCOS::GetFileList(const std::string &dirPath) {
    /*
    This function is getting data from dataset on the path.

    :param dirpath: a string of dataset path
    :return: a collection of file paths

    */
    struct dirent *ptr;
    DIR *dir = opendir(dirPath.c_str());
    std::vector<std::string> files;
    while ((ptr = readdir(dir)) != NULL) {
        if (ptr->d_name[0] == '.') continue;
        files.push_back(ptr->d_name);
    }
    closedir(dir);
    return files;
}

APP_ERROR FCOS::ReadImage(const std::string& imgPath, cv::Mat& imageMat, int& height, int& width) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    width = imageMat.cols;
    height = imageMat.rows;
    
    return APP_ERR_OK;
}

APP_ERROR FCOS::ResizeImage(const cv::Mat& srcImageMat, cv::Mat& dstImageMat) {
    float resizeHeight = 800;
    float resizeWidth = 1333;
    float scale = std::min(resizeWidth / srcImageMat.cols, resizeHeight / srcImageMat.rows);
    int new_width = srcImageMat.cols * scale;
    int new_height = srcImageMat.rows * scale;
    const int average = 2;
    int pad_w = resizeWidth - new_width;
    int pad_h = resizeHeight - new_height;
    int pad_left = pad_w / average;
    int pad_right = pad_w - pad_left;
    int pad_top = pad_h / average;
    int pad_bottom = pad_h - pad_top;
    
    cv::resize(srcImageMat, dstImageMat, cv::Size(new_width,new_height), 0, 0, cv::INTER_CUBIC); //指定常量像素填充
    cv::copyMakeBorder(dstImageMat, dstImageMat, pad_top, pad_bottom, pad_left, pad_right, 
                       cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    return APP_ERR_OK;
}

APP_ERROR FCOS::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase)
{
    const uint32_t dataSize = imageMat.cols * imageMat.rows * YUV444_RGB_WIDTH_NU;
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(imageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imageMat.rows * YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}


APP_ERROR FCOS::Inference(const std::vector<MxBase::TensorBase>& inputs, std::vector<MxBase::TensorBase>& outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR FCOS::PostProcess(const std::string& imgPath, std::vector<MxBase::TensorBase>& inputs, 
                            const std::string &resultPath, int& height, int& width, const std::string& name, 
                            std::string &showPath, float& PROB_THRES) {
    MxBase::TensorBase& tensor = inputs[1]; //1*100
    int ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_1 deploy to host failed.";
        return ret;
    }
    std::vector<uint32_t> shape = tensor.GetShape();
    
    auto  labels = reinterpret_cast<int64 (*)>(tensor.GetBuffer()); //1*100
    
    int label[100] = {0};
    for(int i = 0; i < 100; i++){
        label[i] = labels[i];
    }
   
    tensor = inputs[0]; //1*100*5
    ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_0 deploy to host failed.";
        return ret;
    }
    
    auto bbox = reinterpret_cast<float (*)[5]>(tensor.GetBuffer());
   
    // get infer coordinates
    float image_size_w = width;
    float image_size_h = height;
    float net_input_width = 1333;
    float net_input_height = 800;
    float scale = std::min(net_input_width / image_size_w, net_input_height / image_size_h);

    int pad_w = net_input_width - image_size_w * scale;
    int pad_h = net_input_height - image_size_h * scale;
    int pad_left = pad_w / 2;
    int pad_right = pad_w -pad_left;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h -pad_top;
    
    float prob_thres = PROB_THRES;    
    float ppbox[100][5] = {0}; 
    
    for (int j = 0; j < 100; j++) {
        ppbox[j][0] = (bbox[j][0]-pad_left)/scale;
        ppbox[j][1] = (bbox[j][1]-pad_top)/scale;
        ppbox[j][2] = (bbox[j][2]-pad_right)/scale;
        ppbox[j][3] = (bbox[j][3]-pad_bottom)/scale;
        ppbox[j][4] = bbox[j][4];
        
        if (ppbox[j][0]<0) ppbox[j][0] = 0;
        if (ppbox[j][1]<0) ppbox[j][1] = 0;
        if (ppbox[j][2]>image_size_w) ppbox[j][2] = image_size_w;
        if (ppbox[j][3]>image_size_h) ppbox[j][3] = image_size_h;

    }
    
    std::ofstream out(resultPath);
    cv::Mat imgCur = cv::imread(imgPath);
    for (int j = 0;j<100;j++) {
        if (float(ppbox[j][4])<float(prob_thres)) {
            continue;
        }  
        if (label[j]<0 ||label[j]>80 ) {
            continue;
        }
            
        std::string class_name = classes[int(label[j])];
        std::string det_results_str = "";
        std::ostringstream oss;
        oss<<ppbox[j][4];
        std::string confidence(oss.str());
        
        char strff1[21],strff2[21],strff3[21],strff4[21],strff0[21];
        memset(strff1,0,sizeof(strff1));
        memset(strff2,0,sizeof(strff2));
        memset(strff3,0,sizeof(strff3));
        memset(strff4,0,sizeof(strff4));
        memset(strff0,0,sizeof(strff0));
        // 把浮点数ff转换为字符串，存放在strff中。
        sprintf(strff1,"%.4f",ppbox[j][1]);
        sprintf(strff2,"%.4f",ppbox[j][2]);
        sprintf(strff3,"%.4f",ppbox[j][3]);
        sprintf(strff4,"%.4f",ppbox[j][4]);
        sprintf(strff0,"%.4f",ppbox[j][0]);
        det_results_str = det_results_str+class_name+" "+strff4+" "+strff0+" "+strff1+" "+strff2+" "+strff3+"\n";
        
        LogInfo<<det_results_str;
        out<<det_results_str;
       
        cv::Point p3((ppbox[j][0]), (ppbox[j][1]));
        cv::Point p4((ppbox[j][2]), (ppbox[j][3]));
        cv::Scalar colorRectangle1(0, 255, 1);
        int thicknessRectangle1 = 1;
        cv::rectangle(imgCur, p3, p4, colorRectangle1, thicknessRectangle1);
        cv::putText(imgCur, class_name+"|"+confidence, p3, cv::FONT_HERSHEY_SIMPLEX, 0.5, colorRectangle1,1,1,false);
    }
    out.close();
    cv::imwrite(showPath+"/"+name+".jpg",imgCur);
    return APP_ERR_OK;
}

APP_ERROR FCOS::Process(const std::string &dirPath,  std::string &resultPath,std::string &showPath,float& PROB_THRES) {
    std::vector<std::string> dirFileList = GetFileList(dirPath);
    std::vector<std::string> names, paths;
    int i = 0;
    for (auto imgFile : dirFileList) {
        std::string imgPath = dirPath + "/" + imgFile;
        std::string name = imgFile.substr(0, imgFile.find("."));
        std::string subresultPath = resultPath+"/"+name+".txt";
        cv::Mat imageMat;
        int height, width;
        APP_ERROR ret = ReadImage(imgPath, imageMat, height, width);
        if (ret != APP_ERR_OK) {
            LogError << "ReadImage failed, ret=" << ret << ".";
            return ret;
        }
        ResizeImage(imageMat, imageMat);
        TensorBase tensorBase;
        ret = CVMatToTensorBase(imageMat, tensorBase);
        if (ret != APP_ERR_OK) {
            LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
            return ret;
        }
        std::vector<MxBase::TensorBase> inputs = {};
        std::vector<MxBase::TensorBase> outputs = {};
        inputs.push_back(tensorBase);
        ret = Inference(inputs, outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        ret = PostProcess(imgPath, outputs, subresultPath, height, width,name,showPath,PROB_THRES);
        if (ret != APP_ERR_OK) {
            LogError << "PostProcess failed, ret=" << ret << ".";
            return ret;
        }
        i++;
        LogInfo<<i;
    }
        return APP_ERR_OK;
}
