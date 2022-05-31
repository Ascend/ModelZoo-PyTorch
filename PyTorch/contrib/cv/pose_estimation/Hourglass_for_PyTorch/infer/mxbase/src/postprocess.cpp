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
 
#include "postprocess.h"


APP_ERROR HourglassPostprocess::Init() {
    LogInfo << "Begin to initialize HourglassPostprocess.";
    LogInfo << "End to initialize HourglassPostprocess.";
    return APP_ERR_OK;
}


APP_ERROR HourglassPostprocess::DeInit() {
    LogInfo << "Begin to deinitialize HourglassPostprocess.";
    LogInfo << "End to deinitialize HourglassPostprocess.";
    return APP_ERR_OK;
}


void HourglassPostprocess::GetHeatmap(const std::vector<MxBase::TensorBase>& tensors,
                                    uint32_t heatmapHeight, uint32_t heatmapWeight) {
    auto bboxPtr = reinterpret_cast<float*>(tensors[0].GetBuffer());
    std::shared_ptr<void> keypoint_pointer;
    keypoint_pointer.reset(bboxPtr, floatDeleter);

    for (size_t i = 0; i < NPOINTS; i++) {
        int startIndex = i * heatmapHeight * heatmapWeight;
        for (size_t j = 0; j < heatmapHeight; j++) {
            int middleIndex = j * heatmapWeight;
            for (size_t k = 0; k < heatmapWeight; k++) {
                float x = static_cast<float*>(keypoint_pointer.get())[startIndex + j * heatmapWeight + k];
                heatmaps_reshape[i][j * heatmapWeight + k] = x;
                batch_heatmaps[i][j][k] = x;
            }
        }
    }
}


int HourglassPostprocess::GetIntData(const int index, const float(*heatmaps_reshape)[NUMS_HEAPMAP]) {
    int idx_tem = 0;
    float tem = 0;
    for (int j = 0; j < NUMS_HEAPMAP; j++) {
        if (heatmaps_reshape[index][j] > tem) {
            tem = heatmaps_reshape[index][j];
            idx_tem = j;
        }
    }
    return idx_tem;
}


double HourglassPostprocess::GetFloatData(const int index, const float(*heatmaps_reshape)[NUMS_HEAPMAP]) {
    float tem = 0;
    for (int j = 0; j < NUMS_HEAPMAP; j++) {
        if (heatmaps_reshape[index][j] > tem) {
            tem = heatmaps_reshape[index][j];
        }
    }
    return tem;
}


void HourglassPostprocess::GetAffineMatrix(const float center[], const float scale[], cv::Mat *warp_mat) {
    float scale_tem[2] = {};
    scale_tem[0] = scale[0] * SCALE_RATIO;
    scale_tem[1] = scale[1] * SCALE_RATIO;
    float src_w = scale_tem[0];
    float dst_w = WIDTH_HEAPMAP;
    float dst_h = HEIGHT_HEAPMAP;
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
    cv::Mat warp_mat_af(2, 3, CV_32FC1);
    warp_mat_af = cv::getAffineTransform(dstPoint2f, srcPoint2f);
    *warp_mat = warp_mat_af;
}


void HourglassPostprocess::ParseHeatmap(const std::vector<MxBase::TensorBase>& tensors,
                                            std::vector<float> *preds_result,
                                            uint32_t heatmapHeight, uint32_t heatmapWeight,
                                            const float center[], const float scale[]) {
    LogInfo << "Begin to ParseHeatmap.";
    GetHeatmap(tensors, heatmapHeight, heatmapWeight);
    float maxvals[NPOINTS] = {};
    int idx[NPOINTS] = {};
    for (size_t i = 0; i < NPOINTS; i++) {
        maxvals[i] = GetFloatData(i, heatmaps_reshape);
        idx[i] = GetIntData(i, heatmaps_reshape);
    }
    float preds[NPOINTS][2] = {};
    for (size_t i = 0; i < NPOINTS; i++) {
        preds[i][0] = (idx[i]) % heatmapWeight;
        preds[i][1] = floor(idx[i] / heatmapWeight);
        if (maxvals[i] < 0) {
            preds[i][0] = preds[i][0] * (-1);
            preds[i][1] = preds[i][0] * (-1);
        }
    }
    for (size_t i = 0; i < NPOINTS; i++) {
        float hm[HEIGHT_HEAPMAP][WIDTH_HEAPMAP] = {};
        for (size_t m = 0; m < HEIGHT_HEAPMAP; m++) {
            for (size_t n = 0; n < WIDTH_HEAPMAP; n++) {
                hm[m][n] = batch_heatmaps[i][m][n];
            }
        }
        int px = static_cast<int>(floor(preds[i][0] + 0.5));
        int py = static_cast<int>(floor(preds[i][1] + 0.5));
        if (px > 1 && px < heatmapWeight - 1 && py>1 && py < heatmapHeight - 1) {
            float diff_x = hm[py][px + 1] - hm[py][px - 1];
            float diff_y = hm[py + 1][px] - hm[py - 1][px];
            if (diff_x > 0) {
                preds[i][0] = preds[i][0] + 0.25;
            }
            if (diff_x < 0) {
                preds[i][0] = preds[i][0] - 0.25;
            }
            if (diff_y > 0) {
                preds[i][1] = preds[i][1] + 0.25;
            }
            if (diff_y < 0) {
                preds[i][1] = preds[i][1] - 0.25;
            }
        }
    }
    cv::Mat warp_mat(2, 3, CV_32FC1);
    GetAffineMatrix(center, scale, &warp_mat);
    for (size_t i = 0; i < NPOINTS; i++) {
        preds[i][0] = preds[i][0] * warp_mat.at<double>(0, 0) +
            preds[i][1] * warp_mat.at<double>(0, 1) + warp_mat.at<double>(0, 2);
        preds[i][1] = preds[i][0] * warp_mat.at<double>(1, 0) +
            preds[i][1] * warp_mat.at<double>(1, 1) + warp_mat.at<double>(1, 2);
    }
    for (size_t i = 0; i < NPOINTS; i++) {
        preds_result->push_back(preds[i][0]);
        preds_result->push_back(preds[i][1]);
        preds_result->push_back(maxvals[i]);
    }
}


APP_ERROR HourglassPostprocess::Process(const float center[], const float scale[],
                                        const std::vector<MxBase::TensorBase> &tensors,
                                        std::vector<std::vector<float>>* node_score_list) {
    LogDebug << "Begin to Hourglass PostProcess.";
    auto inputs = tensors;
    APP_ERROR ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed, ret=" << ret;
        return ret;
    }
        
    auto shape = inputs[0].GetShape();
    uint32_t batchSize = shape[0];
    uint32_t heatmapHeight = shape[2];
    uint32_t heatmapWeight = shape[3];
    for (uint32_t i = 0; i < batchSize; ++i) {
        std::vector<float> preds_result;
        ParseHeatmap(inputs, &preds_result, heatmapHeight, heatmapWeight, center, scale);
        node_score_list->push_back(preds_result);
    }    
    LogInfo << "End to Hourglass PostProcess.";
    return APP_ERR_OK;
}

