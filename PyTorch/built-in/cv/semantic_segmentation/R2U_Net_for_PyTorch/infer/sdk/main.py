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

import argparse
import os
import cv2
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from scipy.special import expit
from StreamManagerApi import StreamManagerApi, MxDataInput
from StreamManagerApi import *



##########################evaluation function########################

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)
    corr = np.sum(SR == GT)
    tensor_size = SR.shape[0] * SR.shape[1] * SR.shape[2] * SR.shape[3]
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    # TP : True Positive
    # FN : False Negative
    SR = SR > threshold
    GT = GT == np.max(GT)
    TP = SR & GT
    FN = (~SR) & GT

    SE = float(np.sum(TP)) / (float(np.sum(TP) + np.sum(FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    # TN : True Negative
    # FP : False Positive
    SR = SR > threshold
    GT = GT == np.max(GT)
    TN = (~SR) & (~GT)
    FP = SR & (~GT)

    SP = float(np.sum(TN)) / (float(np.sum(TN) + np.sum(FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    # TP : True Positive
    # FP : False Positive
    SR = SR > threshold
    GT = GT == np.max(GT)
    TP = SR & GT
    FP = SR & (~GT)

    PC = float(np.sum(TP)) / (float(np.sum(TP) + np.sum(FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == np.max(GT)
    Inter = np.sum((SR & GT))
    Union = np.sum((SR | GT))

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == np.max(GT)
    Inter = np.sum((SR & GT))
    DC = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)

    return DC


###########################preprocessing and dataset#############################


def data_processing(image, gt):
    image = cv2.resize(image, (224, 224))
    gt = cv2.resize(gt, (224, 224))
    return image, gt


class ImageFolder(object):
    def __init__(self, root, image_size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_paths.sort()
        self.image_size = image_size
        self.mean = mean
        self.std = std
        print("image count in test path :{}".format(len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('_')[-1][:-len(".jpg")]
        GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        GT = cv2.imread(GT_path, 0)

        image, GT = data_processing(image, GT)
        image = np.array(image).astype(np.float32)
        GT = np.array(GT).astype(np.float32)

        mean = self.mean
        std = self.std
        mean = np.dstack((np.ones((224, 224)) * mean[0] * 255, np.ones((224, 224)) * mean[1] * 255,
                          np.ones((224, 224)) * mean[2] * 255))
        # mean = np.transpose(mean, (2, 0, 1))
        std = np.dstack((np.ones((224, 224)) * std[0] * 255, np.ones((224, 224)) * std[1] * 255,
                         np.ones((224, 224)) * std[2] * 255))
        # std = np.transpose(std, (2, 0, 1))
        image = np.divide(image - mean, std)
        image = np.array(image).astype(np.float32)
        GT = np.array(GT).astype(np.float32)
        image = np.around(image, 6)
        GT = np.around(GT, 6)
        return image, GT

    def __len__(self):
        return len(self.image_paths)


def main(config):
    ##############################train##############################################################
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    config.test_path = os.path.join(config.data_path, "valid")

    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open(config.pipeline_path, 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    data_input = MxDataInput()

    label = []
    predict_result = []
    results_save = []
    label_save = []
    isic_dataset = ImageFolder(root=config.test_path)
    for image, GT in isic_dataset:

        test_image = image
        test_image = np.transpose(test_image, (2, 0, 1))
        test_image = np.expand_dims(test_image, 0)
        GT = np.expand_dims(GT, 0)
        GT = np.expand_dims(GT, 0)
        label.append(GT)
        label_save.append(GT.reshape(-1))
        print(1)
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
        unique_id = stream_manager.SendProtobuf(b'R2U_Net', b'appsrc0', protobufVec1)
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager.GetProtobuf(b'R2U_Net', 0, keyVec)

        if infer_result.size() == 0:
            print("inferResult is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape(1, 1, 224, 224)
        predict_result.append(res)
        res = res.reshape(-1)
        results_save.append(res)

    np.savetxt('test_result.txt', results_save, fmt='%.06f')
    np.savetxt('test_label.txt', label_save, fmt='%.06f')
    print("save result success")

    ############################eval###################################################
    image_size = len(predict_result)
    threshold = 0.5

    acc = 0.  # Accuracy
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    F1 = 0.  # F1 Score
    JS = 0.  # Jaccard Similarity
    DC = 0.  # Dice Coefficient
    length = 0

    for i in range(image_size):
        res = predict_result[i]
        test_GT_image = label[i]

        SR = expit(res)
        SR_ac = SR > threshold
        GT_ac = test_GT_image == np.max(test_GT_image)
        acc += get_accuracy(SR_ac, GT_ac)
        SE += get_sensitivity(SR_ac, GT_ac)
        SP += get_specificity(SR_ac, GT_ac)
        PC += get_precision(SR_ac, GT_ac)
        F1 += get_F1(SR_ac, GT_ac)
        JS += get_JS(SR_ac, GT_ac)
        DC += get_DC(SR_ac, GT_ac)
        length += 1

    acc = acc / length
    SE = SE / length
    SP = SP / length
    PC = PC / length
    F1 = F1 / length
    JS = JS / length
    DC = DC / length
    unet_score = JS + DC
    print("Test finished, model acc: %.3f " % (acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--model_type', type=str, default='R2U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--pipeline_path', type=str, default='../data/config/R2U_Net.pipeline')
    parser.add_argument('--data_path', type=str, default='../data/input/')

    config = parser.parse_args()

    main(config)
