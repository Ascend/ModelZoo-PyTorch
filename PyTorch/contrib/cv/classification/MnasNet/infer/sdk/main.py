#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2021 Huawei Technologies Co., Ltd

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

from StreamManagerApi import *
import os
import cv2
import json
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import time
import sys


def save_infer_result(result, result_name, image_name):
    """
    save the infer result to name_1.txt
    the file content top5:
        class_id1, class_id2, class_id3, class_id4, class_id5
    """
    load_dict = json.loads(result)
    if load_dict.get('MxpiClass') is None:
        with open(result_name + "/" + image_name[:-5] + '.txt', 'w') as f_write:
            f_write.write("")
    else:
        res_vec = load_dict['MxpiClass']
        with open(result_name + "/" + image_name[:-5] + '_1.txt', 'w') as f_write:
            list1 = [str(item.get("classId") - 1) + " " for item in res_vec]
            f_write.writelines(list1)
            f_write.write('\n')


if __name__ == '__main__':
    
    # init stream manager
    stream_manager = StreamManagerApi()
    
    ret = stream_manager.InitManager()
    
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    
    # create streams by pipeline config file
    with open("mnasnet_opencv.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    print(" prepare data_input")
    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2] 

    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    time_start = time.time() 
    for file_name in file_list:
        
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue

        with open(file_path, 'rb') as f:
            data_input.data = f.read()

        
        # Inputs data to a specified stream based on streamName.
        stream_name = b'im_mnasnet'
        inplugin_id = 0
                
        unique_id = stream_manager.SendData(stream_name, inplugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        
        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = stream_manager.GetResult(stream_name, unique_id)
        print(infer_result.data.decode())
        
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        
        # print the infer result
        save_infer_result(infer_result.data.decode(), res_dir_name, file_name)
    end = time.time()
    print("total time:", int(end - time_start))
    print(" time:", int(end - time_start)/len(file_list))
    # destroy streams
    stream_manager.DestroyAllStreams()
