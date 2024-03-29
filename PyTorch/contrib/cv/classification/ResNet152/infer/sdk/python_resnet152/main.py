#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2022 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import datetime
import json
import numpy as np
import os
import sys
import argparse
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput




parser = argparse.ArgumentParser(description='PyTorch resnet152 Training')
parser.add_argument('--pipeline_path', default='../pipeline/resnet152.pipeline', type=str,
                    help='path to pipeline')
parser.add_argument('--dir_name', metavar='DIR', help='path to dataset')
parser.add_argument('--res_dir_name', metavar='DIR', help='path to result')
parser.add_argument('--modelPath', default='../../data/om/resnet152.om', type=str,
                    help='path to model')
parser.add_argument('--ConfigPath', default="../models/resnet152.cfg", type=str,
                   help='path to configpath')
parser.add_argument('--labelPath', default="../../data/imagenet1000_clsidx_to_labels.names", type=str,
                   help='path to labelpath')

if __name__ == '__main__':
    args = parser.parse_args()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    sdk_install_path = os.environ['MX_SDK_HOME']

    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    pipeline_path = args.pipeline_path
    with open(pipeline_path, 'r', encoding='utf-8') as f:
        pipeline_dict = json.load(f)

        pipeline_dict['im_resnet152']["mxpi_tensorinfer0"]['props']['modelPath'] = \
                args.modelPath
        pipeline_dict['im_resnet152']["mxpi_classpostprocessor0"]['props']['postProcessLibPath'] = \
                "libresnet50postprocess.so"
        pipeline_dict['im_resnet152']["mxpi_classpostprocessor0"]['props']['postProcessConfigPath'] = \
                args.ConfigPath
        pipeline_dict['im_resnet152']["mxpi_classpostprocessor0"]['props']['labelPath'] = \
                args.labelPath
    with open(pipeline_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_dict, f, ensure_ascii=False, indent=3)

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # Construct the input of the stream
    data_input = MxDataInput()
    dir_name = args.dir_name
    res_dir_name = args.res_dir_name
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    for file_name in file_list:
        file_path = dir_name + file_name
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue

        with open(file_path, 'rb') as f:
            data_input.data = f.read()
        
        empty_data = []
        stream_name = b'im_resnet152'
        in_plugin_id = 0
        unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.
        start_time = datetime.datetime.now()
        infer_result = stream_manager_api.GetResult(stream_name, unique_id)
        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # print the infer result
        infer_res = infer_result.data.decode()
        print("process img: {}, infer result: {}".format(file_name, infer_res))
        
        load_dict = json.loads(infer_result.data.decode())
        if load_dict.get('MxpiClass') is None:
            with open(res_dir_name + "/" + file_name[:-5] + '.txt', 'w') as f_write:
                f_write.write("")
            continue
        res_vec = load_dict.get('MxpiClass')

        with open(res_dir_name + "/" + file_name[:-5] + '_1.txt', 'w') as f_write:
            res_list = [str(item.get("classId")) + " " for item in res_vec]
            f_write.writelines(res_list)
            f_write.write('\n')

    # destroy streams
    stream_manager_api.DestroyAllStreams()
