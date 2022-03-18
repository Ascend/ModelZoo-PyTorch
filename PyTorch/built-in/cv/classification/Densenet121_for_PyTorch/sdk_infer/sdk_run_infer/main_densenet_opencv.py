#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2020 Huawei Technologies Co., Ltd

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

from StreamManagerApi import StreamManagerApi, MxDataInput
import os
import cv2
import json
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import datetime

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/densenet121_opencv.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = './val_union/'
    res_dir_name = 'densenet121_npu_result'
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
        print(file_name)
        file_path = dir_name + file_name
        if file_name.lower().endswith((".JPEG", ".jpeg", "JPG", "jpg")):
            portion = os.path.splitext(file_name)
            with open(file_path, 'rb') as f:
                data_input.data = f.read()
        else:
            continue
        
        empty_data = []

        stream_name = b'im_densenet121'
        in_plugin_id = 0
        uniqueId = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying stream_name and uniqueId.
        start_time = datetime.datetime.now()
        infer_result = stream_manager_api.GetResult(stream_name, uniqueId)
        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # print the infer result
        print(infer_result.data.decode())
        
        load_dict = json.loads(infer_result.data.decode())
        if load_dict.get('MxpiClass') is None:
            with open(res_dir_name + "/" + file_name[:-5] + '.txt', 'w') as f_write:
                f_write.write("")
            continue
        res_vec = load_dict['MxpiClass']

        with open(res_dir_name + "/" + file_name[:-5] + '_1.txt', 'w') as f_write:
            list1 = [str(item.get("classId")) + " " for item in res_vec]
            f_write.writelines(list1)
            f_write.write('\n')

    # destroy streams
    stream_manager_api.DestroyAllStreams()
