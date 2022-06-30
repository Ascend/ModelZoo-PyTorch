#!/usr/bin/env python
# coding=utf-8

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import argparse
import numpy as np
from preprocess import get_img
from postprocess import get_final_preds
from cal_accuracy import MPIIEval
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn


def parser_args():
    parser = argparse.ArgumentParser(description = 'Params of Hourglass Model:')
    parser.add_argument("--annot_dir", type=str, default="../../data/mpii/annotations/valid.h5")
    parser.add_argument("--img_dir", type=str, default="../../data/mpii/images")

    args = parser.parse_args()
    return args


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for k in tensor.shape:
        tensor_vec.tensorShape.append(k)
    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    rete = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if rete < 0:
        print("Failed to send data to stream.")
        return False
    return True
    

def drawSkeletonLine(imgName, locs):
    '''
    draw skeleton line on orignal image
    '''
    imgPath = "%s/%s" % (args.img_dir, imgName) 
    orgImg = cv2.imread(imgPath)

    for i in range(locs.shape[1]):
        cv2.circle(orgImg, (int(locs[0][i][0]), int(locs[0][i][1])), 3, [0, 255, 85], -1)

    skeleton_line = [[0, 1], [1, 2], [2, 12], [12, 11], [11, 10], [12, 7], [7, 8], [8, 9], 
                    [7, 6], [7, 13], [13, 14], [14, 15], [13, 3], [3, 6], [6, 2], [3, 4], [4, 5]]
    for line_points in skeleton_line:
        x1 = int(locs[0][line_points[0]][0])
        y1 = int(locs[0][line_points[0]][1])
        x2 = int(locs[0][line_points[1]][0])
        y2 = int(locs[0][line_points[1]][1])
        cv2.line(orgImg, (x1, y1), (x2, y2), (255, 0, 0), 3)

    if not os.path.exists("./infer_result"):
        os.mkdir("./infer_result")
    savePath = "%s/%s" % ("./infer_result", imgName)
    cv2.imwrite(savePath, orgImg)


if __name__ == '__main__':
    args = parser_args()
    
    # create stream with pipeline file
    streamManagerApi = StreamManagerApi()   
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open(os.path.realpath("../config/hourglass.pipeline"), 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    
    streamName = b'hourglass'
    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key) 
    
    gts = [] # ground truth labels
    preds = [] # predictions
    normalizing = [] # normalizations used for evaluation

    for kps, img, c, s, n, imgName in get_img(args): # pre-process
        
        # send an image to stream
        if not send_source_data(0, img, streamName, streamManagerApi):
            exit()
        
        # get infer result from stream
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)    
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (infer_result[0].errorCode, infer_result[0].data.decode()))
            exit()            
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        result = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape([1,16,96,96])

        # post-process
        locs, maxvals = get_final_preds(result.copy(), c, s)
        
        # visualization
        drawSkeletonLine(imgName, locs)

        gts.append(kps)
        normalizing.append(n)
        pred = []
        for i in range(locs.shape[0]):
            pred.append({"keypoints": locs[i, :, :]})
        preds.append(pred)
    
    # calculate PCK accuracy
    bound = 0.5 # PCK's threshold of normalized distance
    mpii_eval = MPIIEval()
    mpii_eval.eval(preds, gts, normalizing, bound)
            
    # destroy stream
    streamManagerApi.DestroyAllStreams()