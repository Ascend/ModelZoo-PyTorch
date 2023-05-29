#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import sys
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector

from itertools import islice
import datetime


def send_protobuf(next_batch_inputs, f_ground_truth_out):
    tensors = []
    for batch_input in next_batch_inputs:
        batch_input = np.array(batch_input.strip().split(','))
        dummy_input = batch_input[1:].astype("float32").reshape([1, -1])
        tensors.append(dummy_input)

        f_ground_truth_out.write(str(batch_input[0]) + '\n')

    in_plugin_id = 0
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    for tensor in tensors:
        tensor_package = tensor_package_list.tensorPackageVec.add()

        array_bytes = tensor.tobytes()
        data_input = MxDataInput()
        data_input.data = array_bytes
        tensor_vec = tensor_package.tensorVec.add()
        tensor_vec.deviceId = 0
        tensor_vec.memType = 0
        for i in tensor.shape:
            tensor_vec.tensorShape.append(i)

        tensor_vec.dataStr = data_input.data
        tensor_vec.tensorDataSize = len(array_bytes)

    key = "appsrc{}".format(in_plugin_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager_api.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)

    return ret


if __name__ == '__main__':
    try:
        # annotation files path, "./wdl_input/wdl_infer.txt"
        data_file_path = sys.argv[1]
        # stream pipeline file path, "./pipeline/Wdl.pipeline"
        pipeline_path = sys.argv[2]
        # result files folder, "./result"
        result_path = sys.argv[3]
    except IndexError:
        print("Please enter data files folder | pipeline file path | store result files folder "
              "Such as: python3 main.py ./wdl_input/wdl_infer.txt ./pipeline/Wdl.pipeline ./result")
        exit(1)

    # init stream manager
    stream_name = b'im_wdl'

    # If the batch's inferring is required, set batch_nums > 2; otherwise, batch_nums = 2
    batch_nums = 512

    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, "rb") as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    f_ground_truth_out = open(os.path.join(result_path, "ground_truth.txt"), "w")
    f_infer_result_out = open(os.path.join(result_path, "infer_result.txt"), "w")

    print("Data load from %s" % data_file_path)

    with open(data_file_path, 'r') as f_data_in:
        while True:
            next_batch_inputs = list(islice(f_data_in, 1, batch_nums))
            if not next_batch_inputs:
                break

            ret = send_protobuf(next_batch_inputs, f_ground_truth_out)
            if ret < 0:
                print("Failed to send data to stream.")
                exit()

            key_vec = StringVector()
            key_vec.push_back(b'mxpi_tensorinfer0')

            start_time = datetime.datetime.now()
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            end_time = datetime.datetime.now()
            print('sdk run time: [{}]s'.format((end_time - start_time).seconds))

            if infer_result.size() == 0:
                print("inferResult is null")
                exit()

            if infer_result[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (
                    inferResult[0].errorCode))
                exit()

            results = MxpiDataType.MxpiTensorPackageList()
            results.ParseFromString(infer_result[0].messageBuf)

            result_idx = 0
            try:
                while True:
                    res = np.frombuffer(results.tensorPackageVec[result_idx].tensorVec[0].dataStr, dtype='<f4')
                    print("infer result is: %f" % res[0])
                    f_infer_result_out.write(str(res[0]) + '\n')
                    result_idx += 1
            except IndexError:
                pass

    f_ground_truth_out.close()
    f_infer_result_out.close()

    # destroy streams
    stream_manager_api.DestroyAllStreams()
