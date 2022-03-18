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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"

using namespace MxBase;

// Complete all initialization work. When you are going to 
// use FSAF class, you should invoke this method immediately.
// 
// you need construct InitParam object for Init.
APP_ERROR FSAF::Init(const InitParam& initParam) {
    // initialize device
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    // initialize context
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    
    // load model
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR FSAF::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

// Get all files under a directory.
// Parameters:
//      dirPath: the directory path
// Return: 
//      a vector of filename, including the suffix.
std::vector<std::string> FSAF::GetFileList(const std::string &dirPath) {
    struct dirent *ptr;
    DIR *dir = opendir(dirPath.c_str());
    std::vector<std::string> files;
    while ((ptr = readdir(dir)) != NULL) {
        if (ptr->d_name[0] == '.') {
            continue;
        }
        files.push_back(ptr->d_name);
    }
    closedir(dir);
    return files;
}

// Read image from a image path.
// Parameters:
//      imgPath: string of image path
//      imageMat: opencv Mat object, for storging image as matrix
//      height: int, storge the image height
//      width: int, storge the image width
// Return: 
//      APP_ERROR object, if read image successfully, return APP_ERR_OK.
APP_ERROR FSAF::ReadImage(const std::string& imgPath, cv::Mat& imageMat, int& height, int& width) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);

    // BGR -> RGB
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);

    width = imageMat.cols;
    height = imageMat.rows;
    
    return APP_ERR_OK;
}

// Resize image to fix size.
// We use RightDown padding style.
// Parameters:
//      srcImageMat: source opencv Mat object, image matrix
//      dstImageMat: storge destination opencv Mat object
//      resizedImageInfo: contain infomation about the image, including before and after scaling
//Return:
//      APP_ERROR object, if resize image successfully, return APP_ERR_OK.
APP_ERROR FSAF::ResizeImage(const cv::Mat& srcImageMat, cv::Mat& dstImageMat, 
    MxBase::ResizedImageInfo& resizedImageInfo) {
    float resizeHeight = 800;
    float resizeWidth = 1216;
    float scale = std::min(resizeWidth / srcImageMat.cols, resizeHeight / srcImageMat.rows);
    int new_width = srcImageMat.cols * scale;
    int new_height = srcImageMat.rows * scale;

    // calculate the padding
    int pad_w = resizeWidth - new_width;
    int pad_h = resizeHeight - new_height;

    resizedImageInfo.heightOriginal = srcImageMat.rows;
    resizedImageInfo.heightResize = resizeHeight;
    resizedImageInfo.widthOriginal = srcImageMat.cols;
    resizedImageInfo.widthResize = resizeWidth;
    resizedImageInfo.resizeType = RESIZER_MS_KEEP_ASPECT_RATIO;

    // invoke opencv method to resize and pad
    cv::resize(srcImageMat, dstImageMat, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
    cv::copyMakeBorder(dstImageMat, dstImageMat, 0, pad_h, 0, pad_w, 
        cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return APP_ERR_OK;
}

APP_ERROR FSAF::Normalize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat)
{
    constexpr size_t ALPHA_AND_BETA_SIZE = 3;
    cv::Mat float32Mat;
    srcImageMat.convertTo(float32Mat, CV_32FC3);

    std::vector<cv::Mat> tmp;
    cv::split(float32Mat, tmp);

    const std::vector<double> mean = {103.53, 116.28, 123.675};
    const std::vector<double> std = {57.375, 57.120, 58.395};
    for (size_t i = 0; i < ALPHA_AND_BETA_SIZE; ++i) {
        tmp[i].convertTo(tmp[i], CV_32FC3, 1 / std[i], - mean[i] / std[i]);
    }
    cv::merge(tmp, dstImageMat);
    return APP_ERR_OK;
}

// Convert Mat to Tensor.
// Parameters:
//      imageMat: input image matrix
//      tensorBase: storge image as tensor
// Return:
//       APP_ERROR object, if convert image successfully, return APP_ERR_OK.
APP_ERROR FSAF::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    // calculate the data size: width * height * depth
    const uint32_t dataSize = imageMat.cols * imageMat.rows * imageMat.channels();
    // allocate memory
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(imageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    // get tensor shape
    // std::vector<uint32_t> shape = {imageMat.rows * YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    std::vector<uint32_t> shape = {
        static_cast<uint32_t>(imageMat.rows),
        static_cast<uint32_t>(imageMat.cols),
        static_cast<uint32_t>(imageMat.channels())};

    // tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
    tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT32);    
    return APP_ERR_OK;
}

// Model inference.
// Parameters:
//      inputs: input image tensor
//      outputs: result tensor of inference result
// Return:
//      APP_ERROR object, if convert image successfully, return APP_ERR_OK.
APP_ERROR FSAF::Inference(const std::vector<MxBase::TensorBase>& inputs, std::vector<MxBase::TensorBase>& outputs) {
    auto dtypes = model_->GetOutputDataType();

    // modelDesc_  get the output tensor size through Init()
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

    // infer the result according to the input tensor
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

// Post process for inference result.
// Scale the bbox to the origin image size.
// Parameters:
//      imgPath: input image path
//      inputs: tensor of image after inference
//      resultPath: the path of storaging infer_result
//      height, width: image's height and width
//      name: image name, not including suffix
//      showPath: the path of storaging visualizition result
// Return:
//      APP_ERROR object, if post process image successfully, return APP_ERR_OK.
APP_ERROR FSAF::PostProcess(const std::string& imgPath, std::vector<MxBase::TensorBase>& inputs, 
    const std::string &resultPath, int& height, int& width, const std::string& name, std::string &showPath) {
    // object num
    int tensor_size = 100;

    MxBase::TensorBase& tensor = inputs[1]; // 1*100

    int ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_1 deploy to host failed.";
        return ret;
    }
    std::vector<uint32_t> shape = tensor.GetShape();
    
    auto labels = reinterpret_cast<int64 (*)>(tensor.GetBuffer());  // 1*100

    std::cout << "---------------------------labels---------------------------" << std::endl;
    int label[tensor_size] = {0};
    for(int i = 0; i < tensor_size; i++){
        std::cout << labels[i] << " ";
        label[i] = labels[i];
    }

    tensor = inputs[0]; // 1*100*5
    ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_0 deploy to host failed.";
        return ret;
    }
  
    auto bbox = reinterpret_cast<float (*)[5]>(tensor.GetBuffer());
    std::cout << "\n---------------------------bboxes--------------------------" << std::endl;
    for(int i = 0; i < tensor_size; i++){
       std::cout << bbox[i][0] << ", " << bbox[i][1] << ", " << bbox[i][2] << ", " 
       << bbox[i][3] << ", " << bbox[i][4] << std::endl;
    }
    
    // get infer coordinates
    float image_size_w = width;
    float image_size_h = height;
    float net_input_width = 1216;
    float net_input_height = 800;
    float scale = std::min(net_input_width / (float)width, net_input_height / (float)height);

    int new_width = image_size_w * scale;

    float n_scale = (float)new_width / image_size_w;
   
    // probability threshold and all classes for label
    float prob_thres = 0.05;
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

    // storge bbox after post processing
    float ppbox[tensor_size][5] = {0};
    
    for (int j = 0; j < tensor_size; ++j) {
        // scale bbox
        ppbox[j][0] = (bbox[j][0]) / n_scale;
        ppbox[j][1] = (bbox[j][1]) / n_scale;
        ppbox[j][2] = (bbox[j][2]) / n_scale;
        ppbox[j][3] = (bbox[j][3]) / n_scale;
        ppbox[j][4] = bbox[j][4];
        // limit bbox in a valid range
        ppbox[j][0] = std::max((float)0, ppbox[j][0]);
        ppbox[j][1] = std::max((float)0, ppbox[j][1]); 
        ppbox[j][2] = std::min(image_size_w, ppbox[j][2]);
        ppbox[j][3] = std::min(image_size_h, ppbox[j][3]);
    }
    
    std::ofstream out(resultPath);
    cv::Mat imgCur = cv::imread(imgPath);

    for (int j = 0; j < tensor_size; ++j) {
        if(float(ppbox[j][4]) < float(prob_thres)) {
            continue;
        }
        if(label[j] < 0 || label[j] > 80) {
            continue;
        }

        // standard the output result   
        std::string class_name = classes[int(label[j])];
        std::string det_results_str = "";
        std::ostringstream oss;
        oss << ppbox[j][4];
        std::string confidence(oss.str());
        char strff1[21], strff2[21], strff3[21], strff4[21], strff0[21];
        memset(strff1, 0, sizeof(strff1));
        memset(strff2, 0, sizeof(strff2));
        memset(strff3, 0, sizeof(strff3));
        memset(strff4, 0, sizeof(strff4));
        memset(strff0, 0, sizeof(strff0));
        // print ppbox to char*
        sprintf(strff0, "%.8f", ppbox[j][0]);
        sprintf(strff1, "%.8f", ppbox[j][1]);
        sprintf(strff2, "%.8f", ppbox[j][2]);
        sprintf(strff3, "%.8f", ppbox[j][3]);
        sprintf(strff4, "%.8f", ppbox[j][4]);
        det_results_str = det_results_str + class_name + " " + strff4 + " " + strff0 + " " + strff1 + " " 
                        + strff2 + " " + strff3 + "\n";

        out << det_results_str;
        std::cout << det_results_str;
        // visualization on the origin image
        cv::Point p3((ppbox[j][0]), (ppbox[j][1]));
        cv::Point p4((ppbox[j][2]), (ppbox[j][3]));
        cv::Scalar colorRectangle1(0, 255, 1);
        int thicknessRectangle1 = 1;
        cv::rectangle(imgCur, p3, p4, colorRectangle1, thicknessRectangle1);
        cv::putText(imgCur, class_name + "|" + confidence, p3, cv::FONT_HERSHEY_SIMPLEX, 
            0.5, colorRectangle1, 1, 1, false);
    }
    out.close();
    cv::imwrite(showPath + "/" + name + ".jpg", imgCur);
        
    return APP_ERR_OK;
}

// Primary method for process all images.
APP_ERROR FSAF::Process(const std::string &dirPath,  std::string &resultPath, std::string &showPath) {
    std::vector<std::string> dirFileList = GetFileList(dirPath);
    std::vector<std::string> names, paths;
    // for debug counting
    int i = 0;
    // process every image
    for(auto imgFile : dirFileList) {
        std::string imgPath = dirPath + "/" + imgFile;
        std::string name = imgFile.substr(0, imgFile.find("."));
        std::string subresultPath = resultPath + "/" + name + ".txt";

        cv::Mat imageMat;
        int height = 0;
        int width = 0;
        // get image infomation
    	APP_ERROR ret = ReadImage(imgPath, imageMat, height, width);
    	if (ret != APP_ERR_OK) {
            LogError << "ReadImage failed, ret=" << ret << ".";
            return ret;
        } 
        // resize image and pad it
        ResizedImageInfo resizedImageInfo;
        ResizeImage(imageMat, imageMat, resizedImageInfo);

        // convert image matrix to tensor
        TensorBase tensorBase;
        ret = CVMatToTensorBase(imageMat, tensorBase);
        if (ret != APP_ERR_OK) {
            LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
            return ret;
        }

        std::vector<MxBase::TensorBase> inputs = {};
        std::vector<MxBase::TensorBase> outputs = {};
        inputs.push_back(tensorBase);
        // infer and get output tensor
        ret = Inference(inputs, outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        // post process the bbox to the origin image,
        // and implement visualizition.
        ret = PostProcess(imgPath, outputs, subresultPath, height, width, name, showPath);
        if (ret != APP_ERR_OK) {
            LogError << "PostProcess failed, ret=" << ret << ".";
            return ret;
        }
        // add count
        i++;
        std::cout << i << std::endl;
    }	
    
    return APP_ERR_OK;

}
