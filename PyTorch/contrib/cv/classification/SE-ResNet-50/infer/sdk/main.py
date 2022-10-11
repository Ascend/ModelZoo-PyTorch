#!/usr/bin/env python
# coding=utf-8

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import datetime
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size. """

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def prepocess(img_path):
    input_size = (299, 299)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = Image.open(img_path).convert('RGB')
    img = resize(img, input_size)  # transforms.Resize(256)
    img = np.array(img, dtype=np.float32)
    img = center_crop(img, 224, 224)  # transforms.CenterCrop(224)
    img = img / 255.  # transforms.ToTensor()
    # mean and variance
    img[..., 0] = (img[..., 0] - mean[0]) / std[0]
    img[..., 1] = (img[..., 1] - mean[1]) / std[1]
    img[..., 2] = (img[..., 2] - mean[2]) / std[2]
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    np_img = img.reshape(1, 3, 224, 224)
    return np_img


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/seresnet50.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = sys.argv[1]

    res_dir_name = 'result'
    file_list = os.listdir(dir_name)
    file_list.sort()
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
        print(file_name)
        file_path = dir_name + file_name
        if file_name.lower().endswith((".JPEG", ".jpeg", "JPG", "jpg")):
            img_np = prepocess(file_path)
        else:
            continue
        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionInfo.format = 0
        vision_vec.visionInfo.width = 224
        vision_vec.visionInfo.height = 224
        vision_vec.visionInfo.widthAligned = 224
        vision_vec.visionInfo.heightAligned = 224

        vision_vec.visionData.memType = 0
        vision_vec.visionData.dataStr = img_np.tobytes()
        vision_vec.visionData.dataSize = len(img_np)

        in_plugin_id = 0
        protobuf = MxProtobufIn()
        protobuf.key = b"appsrc0"
        protobuf.type = b'MxTools.MxpiVisionList'
        protobuf.protobuf = vision_list.SerializeToString()
        protobuf_vec = InProtobufVector()

        protobuf_vec.push_back(protobuf)

        # Inputs data to a specified stream based on streamName.
        stream_name = b'im_seresnet50'
        inplugin_id = 0

        # Send data to stream
        unique_id = stream_manager_api.SendProtobuf(stream_name, inplugin_id, protobuf_vec)

        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying stream_name and uniqueId.
        start_time = datetime.datetime.now()

        infer_result = stream_manager_api.GetResult(stream_name, unique_id)
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
