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
# ============================================================================

from abc import abstractproperty
import os
import sys
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector
from itertools import islice
import argparse

parser = argparse.ArgumentParser(description='SDK infer')
parser.add_argument('--pipeline_path', type = str,default = "../data/config/DeepFM.pipeline")
parser.add_argument('--data_file_path',type = str,default ='../data/val/val.txt')

def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.
    Args:
        appsrc_id: an RGB image:the appsrc component number for SendProtobuf
        tensor: the tensor type of the input file
        stream_name: stream Name
        stream_manager:the StreamManagerApi
    Returns:
        bool: send data success or not
    """
    
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    tensor_vec.tensorDataType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    if appsrc_id < 26:
        tensor_vec.tensorDataType = 3
    else:
        tensor_vec.tensorDataType = 0
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    print("key:",key,"tensor:",tensor)
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True


def main():
    """
    read pipeline and do infer
    """
    # init stream manager
    args = parser.parse_args()
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(args.pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return
    stream_name = b'DeepFM'
    predictions = []
    data_file_path = args.data_file_path

    with open(data_file_path, 'r') as f_data_in:
        data_inputs = list (islice(f_data_in, 1, None))
        Lens = len(data_inputs)
        print("there are ",Lens," inputs in total")
        for i ,data_input in enumerate(data_inputs):
            print("processing ", i, "/" , Lens)
            dataList = np.array(data_input.strip().split('\t'))
            infer_input = np.append(dataList[14 : 40], dataList[1 : 14])
            appsrc0 = infer_input[0 : 26].reshape(1 , 26).astype("float32")
            appsrc1 = infer_input[26 : 39].reshape(1 ,1 ,13).astype("float32")
            if not send_source_data(0, appsrc0, stream_name, stream_manager_api):
                return
            if not send_source_data(1, appsrc1, stream_name, stream_manager_api):
                return
            key_vec = StringVector()
            key_vec.push_back(b'mxpi_tensorinfer0')
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            print(infer_result)
            if infer_result.size() == 0:
                print("NO result!")
                return
            if infer_result[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
                return
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
            print("res:",res)
            predictions.append(res.reshape(1, 1))
    print("=======================================")
    print(predictions)
    print(np.shape(predictions))
    # decode and write to file
    f = open('./results.txt', 'w')
    for batch_out in predictions:
        f.write("{:.8f}".format(batch_out[0,0]))
        f.write("\t")
    f.close()
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()
