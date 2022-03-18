# Copyright 2021 Huawei Technologies Co., Ltd
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
import sys
sys.path.append('../..')
import argparse
import os
import random
from StreamManagerApi import StreamManagerApi, MxDataInput
import json
from StreamManagerApi import *
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from PIL import Image, ImageFile
from scipy.special import expit
from evaluation import *
import cv2

def main(config):
    # cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist

    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch
    config.train_path = os.path.join(config.data_path, "train")
    config.valid_path = os.path.join(config.data_path, "valid")
    config.test_path = os.path.join(config.data_path, "test")


    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open("../data/config/Attention_R2U_Net.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    data_input = MxDataInput()

    test_images = np.loadtxt("../data/input/test_images.txt")
    image_size=test_images.shape[0]
    results = []

    for i in range(image_size):
        test_image = test_images[i].reshape(1, 3, 224, 224).astype(np.float32)

        data_input.data = test_image.tobytes()

        tensorPackageList1 = MxpiDataType.MxpiTensorPackageList()
        tensorPackage1 = tensorPackageList1.tensorPackageVec.add()
        tensorVec1 = tensorPackage1.tensorVec.add()
        tensorVec1.deviceId = 0
        tensorVec1.memType = 0
        for t in test_image.shape:
            tensorVec1.tensorShape.append(t)
        tensorVec1.dataStr = data_input.data
        tensorVec1.tensorDataSize = len(test_image.tobytes())
        protobufVec1 = InProtobufVector()
        protobuf1 = MxProtobufIn()
        protobuf1.key = b'appsrc0'
        protobuf1.type = b'MxTools.MxpiTensorPackageList'
        protobuf1.protobuf = tensorPackageList1.SerializeToString()
        protobufVec1.push_back(protobuf1)
        unique_id = stream_manager.SendProtobuf(b'Attention_R2U_Net', b'appsrc0', protobufVec1)
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager.GetProtobuf(b'Attention_R2U_Net', 0, keyVec)

        if infer_result.size() == 0:
            print("inferResult is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape(1,1,224,224)
        res = res.reshape(-1)
        results.append(res)

    result = np.vstack(results)
    np.savetxt('result.txt', results)
    print("save result success")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--test_model_path', type=str, default='./models')
    parser.add_argument('--data_path', type=str, default='./dataset/train/')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--pretrain',  type=int, default=0)
    parser.add_argument('--pretrain_path',  type=str, default="")
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument('--use_apex', type=int, default=1)
    parser.add_argument('--apex_level', type=str, default="O2")
    parser.add_argument('--loss_scale', type=float, default=128.)


    config = parser.parse_args()




    main(config)
