# coding=utf-8
#
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
import base64
import json
import os

import numpy as np
import cv2 as cv
from glob import glob

import MxpiDataType_pb2 as mxpi_data
from StreamManagerApi import InProtobufVector
from StreamManagerApi import MxProtobufIn
from StreamManagerApi import StreamManagerApi


class MultiClassLoader:
    _plugin_name = "multiclass"

    def __init__(self, dataset_dir):
        super(MultiClassLoader, self).__init__()
        self._dataset_dir = dataset_dir

    def iter_dataset(self):
        for image_id in self.list_image_id():
            image, mask = self.get_image_mask(image_id)
            yield image_id, image, mask

    def list_image_id(self):
        img_ids = glob(os.path.join(self._dataset_dir, 'images', '*.png'))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        return img_ids
        
    def get_image_mask(self, image_id):
        image_dir = os.path.join(self._dataset_dir, 'images')
        image = cv.imread(os.path.join(image_dir, image_id+".png"))
        mask_dir = os.path.join(self._dataset_dir, 'masks/0')
        mask = cv.imread(os.path.join(mask_dir, image_id+".png"), cv.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to get image by id {image_id}")
            
        mask = mask.astype('float32') / 255

        return image, mask

    @classmethod
    def get_plugin_name(cls):
        return cls._plugin_name


class SDKInferWrapper:
    def __init__(self):
        self._stream_name = None
        self._stream_mgr_api = StreamManagerApi()

        if self._stream_mgr_api.InitManager() != 0:
            raise RuntimeError("Failed to init stream manager.")

    def load_pipeline(self, pipeline_path):
        with open(pipeline_path, 'r') as f:
            pipeline = json.load(f)

        self._stream_name = list(pipeline.keys())[0].encode()
        if self._stream_mgr_api.CreateMultipleStreams(
                json.dumps(pipeline).encode()) != 0:
            raise RuntimeError("Failed to create stream.")

    def do_infer(self, image):
        tensor_pkg_list = mxpi_data.MxpiTensorPackageList()
        tensor_pkg = tensor_pkg_list.tensorPackageVec.add()
        tensor_vec = tensor_pkg.tensorVec.add()
        tensor_vec.deviceId = 0
        tensor_vec.memType = 0

        for dim in [1, *image.shape]:
            tensor_vec.tensorShape.append(dim)

        input_data = image.tobytes()
        tensor_vec.dataStr = input_data
        tensor_vec.tensorDataSize = len(input_data)

        protobuf_vec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = b'appsrc0'
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensor_pkg_list.SerializeToString()
        protobuf_vec.push_back(protobuf)

        unique_id = self._stream_mgr_api.SendProtobuf(
            self._stream_name, 0, protobuf_vec)

        if unique_id < 0:
            raise RuntimeError("Failed to send data to stream.")

        infer_result = self._stream_mgr_api.GetResult(
            self._stream_name, unique_id)

        if infer_result.errorCode != 0:
            raise RuntimeError(
                f"GetResult error. errorCode={infer_result.errorCode}, "
                f"errorMsg={infer_result.data.decode()}")
        return infer_result


def _parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="path of dataset directory")
    parser.add_argument("--pipeline", type=str, required=True,
                        help="path of pipeline file")
    parser.add_argument("--output_dir", type=str, default="./infer_result",
                        help="path of output directory")
    return parser.parse_args()


def _parse_output_data(output_data):
    infer_result_data = json.loads(output_data.data.decode())
    content = json.loads(infer_result_data['metaData'][0]['content'])
    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][0]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    infer_array = np.frombuffer(base64.b64decode(data_str), dtype=np.float32)
    return infer_array.reshape(tensor_shape)

def sigmoid(x):
    y = x.copy()
    y[x >= 0] = 1.0 / (1 + np.exp(-x[x >= 0]))
    y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
    return y
    
def iou_score(output, target):
    smooth = 1e-5

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def main():
    args = _parser_args()
    sdk_infer = SDKInferWrapper()
    sdk_infer.load_pipeline(args.pipeline)
    data_loader = MultiClassLoader(args.dataset_dir)
    
    count = 0
    iou_sum = 0
    for image_id, image, mask in data_loader.iter_dataset():
        output_data = sdk_infer.do_infer(image)
        output_tensor = _parse_output_data(output_data)
        os.makedirs(args.output_dir, exist_ok=True)
        tensor = sigmoid(output_tensor[0][0])
        iou = iou_score(tensor, mask)
        print("The image " + str(image_id) + "'s IOU is " + str(iou))
        iou_sum += iou
        count += 1
        filename = os.path.join(args.output_dir,
                                image_id + '.png')
        cv.imwrite(filename,
                   (tensor*255).astype(np.uint8))
    print(f"The Mean IOU is {iou_sum/count}")


if __name__ == "__main__":
    main()