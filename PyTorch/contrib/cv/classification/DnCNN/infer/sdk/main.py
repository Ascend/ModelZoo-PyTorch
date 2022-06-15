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
import sys
import cv2
import numpy as np
from StreamManagerApi import StreamManagerApi, StringVector, MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType

image_size = 481
shape = [1, 1, image_size, image_size]


def info(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("[INFO][%s %d %s:%s] %s" %(nowtime, os.getpid(), __file__, sys._getframe().f_back.f_lineno, msg))

def warn(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("\033[33m[WARN][%s %d %s:%s] %s\033[0m" %(nowtime, os.getpid(), __file__, sys._getframe().f_back.f_lineno, msg))

def err(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("\033[31m[ERROR][%s %d %s:%s] %s\033[0m" %(nowtime, os.getpid(), __file__, sys._getframe().f_back.f_lineno, msg))


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        err("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/DnCNN.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        err("Failed to create Stream, ret=%s" % str(ret))
        exit()

    dir_name = sys.argv[1]
    file_list = os.listdir(dir_name)

    avg_psnr = 0.

    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        info("Read data from %s" % file_path)

        # read grayscale
        img = cv2.imread(file_path, 0)
        img = cv2.resize(img, (image_size, image_size))
        img = np.float32(img[:, :]) / 255

        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 1)

        # Add noise
        noise = np.random.normal(0, 15 / 255., img.shape).astype('float32')
        noisy = img + noise

        # Construct the input of the stream
        tensorPackageList1 = MxpiDataType.MxpiTensorPackageList()
        tensorPackage1 = tensorPackageList1.tensorPackageVec.add()
        tensorVec1 = tensorPackage1.tensorVec.add()
        tensorVec1.deviceId = 0
        tensorVec1.memType = 0
        for t in shape:
            tensorVec1.tensorShape.append(t)

        tensorVec1.dataStr = noisy.tobytes()
        tensorVec1.tensorDataSize = len(tensorVec1.dataStr)

        protobuf1 = MxProtobufIn()
        protobuf1.key = b'appsrc0'
        protobuf1.type = b'MxTools.MxpiTensorPackageList'
        protobuf1.protobuf = tensorPackageList1.SerializeToString()

        protobufVec1 = InProtobufVector()
        protobufVec1.push_back(protobuf1)

        unique_id = stream_manager.SendProtobuf(b'DnCNN', b'appsrc0', protobufVec1)

        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager.GetProtobuf(b'DnCNN', 0, keyVec)
        if infer_result.size() == 0:
            err("inferResult is null")
            exit()

        if infer_result[0].errorCode != 0:
            err("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()
        # get infer result
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        # convert the inference result to Numpy array
        residual = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr,
                dtype=np.float32).reshape(shape)
        denoised = np.clip(noisy - residual, 0., 1.)
        mse = np.mean((img - denoised) ** 2)

        psnr = 10 * np.log10(1. / mse)
        avg_psnr += psnr

        info('psnr value: %.3f' % (psnr))

    avg_psnr /= len(file_list)
    info('final psnr value: %.3f' % (avg_psnr))

    end_time = datetime.datetime.now()
    info('sdk run time: {}us'.format((end_time - start_time).microseconds))
