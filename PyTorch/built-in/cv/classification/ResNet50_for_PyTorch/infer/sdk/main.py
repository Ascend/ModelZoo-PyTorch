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

# import StreamManagerApi.py
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput
import os
import json
import numpy as np
import datetime
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
            list1 = [str(item.get("classId")) + " " for item in res_vec]
            f_write.writelines(list1)
            f_write.write('\n')
            
def main():
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/Resnet50.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)


    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()
    
    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2]
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        if file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg"):
            portion = os.path.splitext(file_name)
            with open(file_path, 'rb') as f:
                data_input.data = f.read()
        else:
            continue
        
        empty_data = []

        stream_name = b'resnet50_classification'
        in_plugin_id = 0
        unique_id = stream_manager_api.SendDataWithUniqueId(stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying stream_name and unique_id.
        start_time = datetime.datetime.now()
        infer_result = stream_manager_api.GetResultWithUniqueId(stream_name, unique_id, 3000)
        endtime = datetime.datetime.now()
        print('sdk run time: {}'.format((endtime - start_time).microseconds))
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # print the infer result
        infer_res = infer_result.data.decode()
        print("process img: {}, infer result: {}".format(file_name, infer_res))
        
        save_infer_result(infer_result.data.decode(), res_dir_name, file_name)

    # destroy streams
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    main()