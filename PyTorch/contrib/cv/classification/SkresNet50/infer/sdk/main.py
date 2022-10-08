# !/usr/bin/env python

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
# ============================================================================

import datetime
import json
import os
import sys
import numpy as np
import MxpiDataType_pb2 as MxpiDataType

from PIL import Image
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput
from StreamManagerApi import StringVector, MxProtobufIn, InProtobufVector


def resize(img, size, interpolation=Image.BILINEAR):
    if img.height <= img.width:
        ratio = size / img.height
        w_size = int(img.width * ratio)
        img = img.resize((w_size, size), interpolation)
    else:
        ratio = size / img.width
        h_size = int(img.height * ratio)
        img = img.resize((size, h_size), interpolation)

    return img


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def run():
    """
        read pipeline and do infer
    """
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open("../data/config/sk_resnet50_npu_16.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2]
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(
                ".jpg") or file_name.lower().endswith(".jpeg")):
            return

        # image preprocess
        input_size = 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = Image.open(file_path).convert('RGB')
        # normalize
        img = resize(img, input_size)  # transforms.Resize(256)
        img = np.array(img, dtype=np.float32)
        img = center_crop(img, 224, 224)   # transforms.CenterCrop(224)
        img = img / 255.  # transforms.ToTensor()

        img[..., 0] = (img[..., 0] - mean[0]) / std[0]
        img[..., 1] = (img[..., 1] - mean[1]) / std[1]
        img[..., 2] = (img[..., 2] - mean[2]) / std[2]

        img = img.transpose(2, 0, 1)   # HWC -> CHW

        stream_name = b'im_sk_resnet50_npu_16'
        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionInfo.format = 0
        vision_vec.visionInfo.width = 224
        vision_vec.visionInfo.height = 224
        vision_vec.visionInfo.widthAligned = 224
        vision_vec.visionInfo.heightAligned = 224

        vision_vec.visionData.memType = 0
        vision_vec.visionData.dataStr = img.tobytes()
        vision_vec.visionData.dataSize = len(img)

        protobuf = MxProtobufIn()
        protobuf.key = b"appsrc0"
        protobuf.type = b'MxTools.MxpiVisionList'
        protobuf.protobuf = vision_list.SerializeToString()
        protobuf_vec = InProtobufVector()

        protobuf_vec.push_back(protobuf)

        # Inputs data to a specified stream based on streamName.
        inplugin_id = 0

        # Send data to stream
        unique_id = stream_manager_api.SendProtobuf(stream_name, inplugin_id, protobuf_vec)
        if unique_id < 0:
            print("Failed to send data to stream.")
            return

        # Obtain the inference result by specifying streamName and uniqueId.
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
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


if __name__ == '__main__':
    run()
