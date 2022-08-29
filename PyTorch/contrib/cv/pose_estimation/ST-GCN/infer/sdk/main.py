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
""" main.py """
import datetime
import os
import pickle
import sys
import argparse
import json

import numpy as np
from StreamManagerApi import StreamManagerApi, StringVector, MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="albert process")
    parser.add_argument("--pipeline", type=str, default="",
                        help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Dataset contain input_ids, input_mask, segment_ids, label_ids")
    parser.add_argument("--label_dir", type=str,
                        default="", help="label ids to name")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    args_opt = parser.parse_args()
    return args_opt


def send_source_data(appsrc_id, filename, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor = np.fromfile(filename, dtype=np.float32)
    tensor = np.expand_dims(tensor, 0)
    tensor = np.reshape(tensor, (1,3,150,18,2))
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
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

    key = "appsrc{}".format(appsrc_id).encode('utf-8')
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


def info(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("[INFO][%s %d %s:%s] %s" %(nowtime, os.getpid(), __file__, sys._getframe().f_back.f_lineno, msg))

def warn(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("\033[33m[WARN][%s %d %s:%s] %s\033[0m" %(nowtime, os.getpid(), __file__, sys._getframe().f_back.f_lineno, msg))

def err(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("\033[31m[ERROR][%s %d %s:%s] %s\033[0m" %(nowtime, os.getpid(), __file__, sys._getframe().f_back.f_lineno, msg))


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy =sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


def run():
    """
    read pipeline and do infer
    """
    args = parse_args()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'im_stgcn'
    # input_ids file list, every file content a tensor[1,128]
    file_list = os.listdir(args.data_dir)
    file_list.sort(key=lambda x:int(x[:-4]))

    fr=open(args.label_dir, 'rb')

    results = []
    objcnt = len(file_list)
    cls_cnts = 400

    for file_name in file_list:
        file_name = os.path.realpath(os.path.join(
        args.data_dir, file_name))

        if not send_source_data(0, file_name, stream_name, stream_manager_api):
            return False

        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" %
                  (infer_result[0].errorCode))
            return

        
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(
        result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='float32')
        results.append(res)


    results = np.concatenate(results)
    results = results.reshape(objcnt, cls_cnts)
   

    ranks = results.argsort()
    infer_result = []
    for rank in ranks:
        infer_result.append(rank[-5:])
    print("infer_results ",infer_result)
    file = open('infer_result.txt', 'w')
    file.write(str(infer_result))
    file.close()

    # read labels
    labels_file = pickle.load(fr)
    labels = labels_file[1]
    labels = labels[:objcnt] # the shortcut of labels
    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))


if __name__ == '__main__':
    # run()
    run()
