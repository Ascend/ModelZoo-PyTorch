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

#include "hourglass.h"


APP_ERROR Hourglass::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    res = initParam.res;

    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
	hourglassPostprocess = std::make_shared<HourglassPostprocess>();
	ret = hourglassPostprocess->Init();
	if (ret != APP_ERR_OK) {
        LogError << "HourglassPostprocess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR Hourglass::DeInit() {
    model_->DeInit();
	hourglassPostprocess->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR Hourglass::ReadImage(const std::string &imgPath, cv::Mat &imageMat, ImageShape &imgShape) {
    cv::Mat bgrImageMat;
    bgrImageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(bgrImageMat, imageMat, cv::COLOR_BGR2RGB);
	imgShape.width = imageMat.cols;
    imgShape.height = imageMat.rows;
    LogInfo << "ImageSize:" << imageMat.size();
    LogInfo << "ImageDims:" << imageMat.dims;
    LogInfo << "ImageChannels:" << imageMat.channels();
    return APP_ERR_OK;
}


APP_ERROR Hourglass::Resize_Affine(const cv::Mat& srcImage, cv::Mat *dstImage, ImageShape *imgShape,
                                    const float center[], const float scale[]) {
    int new_width, new_height;
    new_height = static_cast<int>(imgShape->height);
    new_width = static_cast<int>(imgShape->width);

    float scale_tem[2] = {};
    scale_tem[0] = scale[0] * 200.0;
    scale_tem[1] = scale[1] * 200.0;
    float src_w = scale_tem[0];
    float dst_w = MODEL_WIDTH;
    float dst_h = MODEL_HEIGHT;
    float src_dir[2] = {};
    float dst_dir[2] = {};
    float sn = sin(0);
    float cs = cos(0);
    src_dir[0] = src_w * 0.5 * sn;
    src_dir[1] = src_w * (-0.5) * cs;
    dst_dir[0] = 0;
    dst_dir[1] = dst_w * (-0.5);

    float src[3][2] = {};
    float dst[3][2] = {};

    src[0][0] = center[0];
    src[0][1] = center[1];
    src[1][0] = center[0] + src_dir[0];
    src[1][1] = center[1] + src_dir[1];
    dst[0][0] = dst_w * 0.5;
    dst[0][1] = dst_h * 0.5;
    dst[1][0] = dst_w * 0.5 + dst_dir[0];
    dst[1][1] = dst_h * 0.5 + dst_dir[1];

    float src_direct[2] = {};
    src_direct[0] = src[0][0] - src[1][0];
    src_direct[1] = src[0][1] - src[1][1];
    src[2][0] = src[1][0] - src_direct[1];
    src[2][1] = src[1][1] + src_direct[0];

    float dst_direct[2] = {};
    dst_direct[0] = dst[0][0] - dst[1][0];
    dst_direct[1] = dst[0][1] - dst[1][1];
    dst[2][0] = dst[1][0] - dst_direct[1];
    dst[2][1] = dst[1][1] + dst_direct[0];
    cv::Point2f srcPoint2f[3], dstPoint2f[3];
    srcPoint2f[0] = cv::Point2f(static_cast<float>(src[0][0]), static_cast<float>(src[0][1]));
    srcPoint2f[1] = cv::Point2f(static_cast<float>(src[1][0]), static_cast<float>(src[1][1]));
    srcPoint2f[2] = cv::Point2f(static_cast<float>(src[2][0]), static_cast<float>(src[2][1]));
    dstPoint2f[0] = cv::Point2f(static_cast<float>(dst[0][0]), static_cast<float>(dst[0][1]));
    dstPoint2f[1] = cv::Point2f(static_cast<float>(dst[1][0]), static_cast<float>(dst[1][1]));
    dstPoint2f[2] = cv::Point2f(static_cast<float>(dst[2][0]), static_cast<float>(dst[2][1]));
    cv::Mat warp_mat(2, 3, CV_32FC1);
    warp_mat = cv::getAffineTransform(srcPoint2f, dstPoint2f);

    cv::Mat src_cv(new_height, new_width, CV_8UC3, srcImage.data);

    cv::Mat warp_dst = cv::Mat::zeros(cv::Size(static_cast<int>(MODEL_WIDTH), 
                                        static_cast<int>(MODEL_HEIGHT)), src_cv.type());

    cv::warpAffine(src_cv, warp_dst, warp_mat, warp_dst.size());

    cv::Mat image_finally(warp_dst.rows, warp_dst.cols, CV_32FC3);

    warp_dst.convertTo(image_finally, CV_32FC3, 1 / 255.0);

	float mean[3] = { 0.485, 0.456, 0.406 };
    float std[3] = { 0.229, 0.224, 0.225 };

    for (int i = 0; i < image_finally.rows; i++) {
        for (int j = 0; j < image_finally.cols; j++) {
            if (warp_dst.channels() == 3) {
                image_finally.at<cv::Vec3f>(i, j)[0]= (image_finally.at<cv::Vec3f>(i, j)[0] - mean[0]) / std[0];
                image_finally.at<cv::Vec3f>(i, j)[1]= (image_finally.at<cv::Vec3f>(i, j)[1] - mean[1]) / std[1];
                image_finally.at<cv::Vec3f>(i, j)[2]= (image_finally.at<cv::Vec3f>(i, j)[2] - mean[2]) / std[2];
            }
        }
    }
    *dstImage = image_finally;
    return APP_ERR_OK;
}



APP_ERROR Hourglass::CVMatToTensorBase(const cv::Mat& imageMat, MxBase::TensorBase *tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }
    // mat NHWC to NCHW
    size_t  H = 384, W = 384, C = 3;
    float mat_data[dataSize] = {};
    dataSize = dataSize * 4;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                int id = c * (H * W) + h * W + w;
                mat_data[id] = imageMat.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(mat_data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = { 1, 3, 384, 384 };
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}


APP_ERROR Hourglass::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;

    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR Hourglass::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                    std::vector<std::vector<float> >* node_score_list, 
								    const float center[], const float scale[]) {
    APP_ERROR ret = hourglassPostprocess->Process(center, scale, inputs, node_score_list);
    if (ret != APP_ERR_OK) {
        LogError << "Hourglass Postprocess failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


void drawSkeletonLine(const std::string& imgPath, const std::string& resultPath, const float (*preds)[2]) {
    cv::Mat imageMat = cv::imread(imgPath); // read orignal image
     
    // Draw keypoints
    for (int i = 0; i < NUMS_JOINTS; i++) {
        cv::Point center;
        center.x = static_cast<int>(preds[i][0]);
        center.y = static_cast<int>(preds[i][1]);
        cv::circle(imageMat, center, 4, cv::Scalar(0, 255, 85), -1);
    }
    
    // draw skeleton lines
    std::vector<std::vector<int>> skeletonLines = {{0, 1}, {1, 2}, {2, 12}, {12, 11}, {11, 10}, {12, 7}, 
                                                    {7, 8}, {8, 9}, {7, 6}, {7, 13}, {13, 14}, {14, 15},
                                                    {13, 3}, {3, 6}, {6, 2}, {3, 4}, {4, 5}};
    for (int i = 0; i < skeletonLines.size(); i++) {
        int x1 = int(preds[skeletonLines[i][0]][0]);
        int y1 = int(preds[skeletonLines[i][0]][1]);
        int x2 = int(preds[skeletonLines[i][1]][0]);
        int y2 = int(preds[skeletonLines[i][1]][1]);
        cv::line(imageMat, cv::Point(x1, y1), cv::Point(x2, y2), (255, 0, 0), 3);
    }

    cv::imwrite(resultPath, imageMat); // save image
}


void VisualizeInferResult(const std::string& imgPath, const std::string &resultPath,
                            const std::vector<std::vector<float> >& node_score_list) {
    for (int i = 0; i < node_score_list.size(); i++) {
        float preds[NUMS_JOINTS][2] = {};
        float maxvals[NUMS_JOINTS] = {};
        int idx = 0;
        for (int j = 0; j < node_score_list[i].size(); j += 3) {
            preds[idx][0] = node_score_list[i][j];
            preds[idx][1] = node_score_list[i][j + 1];
            maxvals[idx] = node_score_list[i][j + 2];
            idx++;
        }
        LogInfo << "infer result:";
        LogInfo << "preds:";
        for (int m = 0; m < NUMS_JOINTS; m++) {
            LogInfo << preds[m][0] << "  " << preds[m][1];
        }
        drawSkeletonLine(imgPath, resultPath, preds);
    }
}


APP_ERROR Hourglass::Process(const ImageAnnot &imageAnnot) {
    cv::Mat imageMat;
	ImageShape imageShape{};
    std::string imagePath = "../../data/mpii/images/"; // 数据集图片目录
    APP_ERROR ret = ReadImage(imagePath + imageAnnot.imageName, imageMat, imageShape);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    float center[2] = {imageAnnot.center[0], imageAnnot.center[1]};
    float scale[2] = {imageAnnot.scale[0], imageAnnot.scale[1]};
    cv::Mat dstImage;
    Resize_Affine(imageMat, &dstImage, &imageShape, center, scale);
	
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(dstImage, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);
	
	ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Inference success, ret=" << ret << ".";
	
	std::vector<std::vector<float> > node_score_list = {};
    ret = PostProcess(outputs, &node_score_list, center, scale);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    VisualizeInferResult(imagePath + imageAnnot.imageName, "./result/" + imageAnnot.imageName, node_score_list);
	
    return APP_ERR_OK;
}
