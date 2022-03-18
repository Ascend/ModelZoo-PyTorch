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

import cv2
import datetime
import json
import numpy as np
import os

from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/psenet_dvpp.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = './icdar2015/'
    res_dir_name = 'icdar2015_psenet_npu_result'
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
        starttime = datetime.datetime.now()
        file_path = dir_name + file_name
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue

        with open(file_path, 'rb') as f:
            data_input.data = f.read()
        im = cv2.imread(file_path)
        # Inputs data to a specified stream based on streamName.
        stream_name = b'classification+detection'
        in_plugin_id = 0
        unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = stream_manager_api.GetResult(stream_name, unique_id)
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # print the infer result
        print(infer_result.data.decode())

        load_dict = json.loads(infer_result.data.decode())
        if load_dict.get('MxpiTextObject') is None:
            with open(res_dir_name + "/" + file_name[:-4] + '.txt', 'w') as f_write:
                f_write.write("")
            continue
        res_vec = load_dict.get('MxpiTextObject')
        boxes = []
        for res in res_vec:
            boxes.append([int(res.get('x0')), int(res.get('y0')), int(res.get('x2')), int(res.get('y2'))])
        output_file = res_dir_name + "/" + 'res_' + file_name
        boxes = np.array(boxes, dtype=float)
        for i, box in enumerate(boxes):
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), thickness=2)
        cv2.imwrite(output_file, im)

        with open(res_dir_name + "/" + file_name[:-4] + '.txt', 'w') as f_write:
            for i, box in enumerate(res_vec):
                for ele in ["x3", "y3", "x0", "y0", "x1", "y1", "x2", "y2"]:
                    if box.get(ele) < 0:
                        box[ele] = 0

                f_write.write('{},{},{},{},{},{},{},{}\r\n'.format(box.get("x3"), box.get("y3"), box.get("x0"),
                                                                   box.get("y0"), box.get("x1"), box.get("y1"),
                                                                   box.get("x2"), box.get("y2")))
        endtime = datetime.datetime.now()
        print('sdk run time: {}'.format((endtime - starttime).microseconds))

    # destroy streams
    stream_manager_api.DestroyAllStreams()
