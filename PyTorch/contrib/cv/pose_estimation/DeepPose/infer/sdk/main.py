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
import os.path as osp
import cv2
import argparse
import numpy as np
import MxpiDataType_pb2 as MxpiDataType

from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn
from preprocess import get_img,load_coco_person_detection_results, get_mapping_id_name, data_cfg
from postprocess import get_final_preds,process_result,fliplr_regression
from cal_accuracy import evaluate
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('--data_root',default = '/home/dataset/coco2017' ,help='path of root')
    parser.add_argument('--out', default = './output', help='output result file')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
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


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
    

if __name__ == '__main__':
    args = parse_args()
    # create stream with pipeline file
    streamManagerApi = StreamManagerApi()   
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open(os.path.realpath("../config/DeepPose.pipeline"), 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    streamName = b'deeppose'
    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key) 
    mkdir_or_exist(osp.abspath(args.out))  
    print(args.data_root)

    ann_file = f'{args.data_root}/annotations/person_keypoints_val2017.json'
    coco = COCO(ann_file)
    id2name, name2id = get_mapping_id_name(coco.imgs)
    outputs = [] # predictions

    for img, c, s, img_name, score, bbox_ids in get_img(args, coco, id2name): # pre-process
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
        result = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape([1,17,2])
        
        # send an image_flp to stream
        img_flip = np.flip(img,3)
        if not send_source_data(0, img_flip, streamName, streamManagerApi):
            exit()
        # get infer result from stream
        infer_result_flip = streamManagerApi.GetProtobuf(streamName, 0, keyVec)    
        if infer_result_flip.size() == 0:
            print("infer_result_flip is null")
            exit()
        if infer_result_flip[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (infer_result_flip[0].errorCode, infer_result_flip[0].data.decode()))
            exit()            
        result_flip = MxpiDataType.MxpiTensorPackageList()
        result_flip.ParseFromString(infer_result_flip[0].messageBuf)
        result_flip = np.frombuffer(result_flip.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape([1,17,2])
        # post-process
        result_flip = fliplr_regression(result_flip, data_cfg['flip_pairs'])
        output_heatmap = (result + result_flip) * 0.5
        output = process_result(output_heatmap, c, s, data_cfg['image_size'], score, bbox_ids, img_name)
        outputs.append(output)
    print (outputs[0]['image_paths'])
    img = cv2.imread(outputs[0]['image_paths'])
    for i in range(17):
        x=int(outputs[0]['preds'][0][i][0])
        y=int(outputs[0]['preds'][0][i][1])
        score=outputs[0]['preds'][0][i][2]
        if x>0 and y>0 and score>0.3:
            pos=(x,y)
            cv2.circle(img, pos, 5, color=(0, 255, 0))
    img_name = 'out.jpg'
    cv2.imwrite(img_name, img)
    
    # calculate accuracy
    results = evaluate(outputs, args.out, data_cfg, name2id, coco, args)
    for k, v in sorted(results.items()):
        print(f'{k}: {v}')
    # destroy stream
    streamManagerApi.DestroyAllStreams()
