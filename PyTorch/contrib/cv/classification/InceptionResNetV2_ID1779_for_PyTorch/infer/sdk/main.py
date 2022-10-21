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

import os
import json
import sys
import time
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from PIL import Image
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector


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

    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)

    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')

    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()

    protobuf_vec = InProtobufVector()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """

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


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def cre_groundtruth_dict_fromtxt(gtfile_path):
    img_gt_dict = {}
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            img_name = temp[0].split(".")[0]
            img_lab = temp[1]
            img_gt_dict[img_name] = img_lab
    return img_gt_dict


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
    with open('../data/config/inception_resnet_v2.pipeline', 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'im_iception_resnet_v2'

    # Construct the input of the stream
    infer_total_time = 0
    # image_input_dir, label_input, res_dir
    image_input_dir = sys.argv[1]
    real_label_name = sys.argv[2]
    res_dir_name = sys.argv[3]
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []

    file_list = os.listdir(image_input_dir)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img_gt_dict = cre_groundtruth_dict_fromtxt(real_label_name)
    topn = 5
    count_hit = np.zeros(topn)
    count = 0
    resCnt = 0
    n_labels = 0

    for file_name in file_list:
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue
        count += 1
        file_path = os.path.join(image_input_dir, file_name)
        # preprocess
        img = Image.open(file_path).convert('RGB')
        # transforms.Resize(342)
        img = resize(img, 342)
        img = np.array(img, dtype=np.float32)
        # transforms.CenterCrop(229, 229)
        img = center_crop(img, 299, 299)
        # transforms.ToTensor()
        img = img / 255.
        # mean and variance
        img[..., 0] = (img[..., 0] - mean[0]) / std[0]
        img[..., 1] = (img[..., 1] - mean[1]) / std[1]
        img[..., 2] = (img[..., 2] - mean[2]) / std[2]

        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # reshape(1, 3, 299, 299)
        # tensor = img.reshape(1, 3, 299, 299)
        tensor = np.expand_dims(img, 0)
        print('tensor_shape: ', tensor.shape)

        # send_source_data
        if not send_source_data(0, tensor, stream_name, stream_manager_api):
            return

        # Obtain the inference result by specifying streamName and uniqueId.
        # GetProtobuf
        in_plugin_id = 0
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(
            stream_name, in_plugin_id, key_vec)

        infer_total_time += time.time() - start_time

        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" %
                  (infer_result[0].errorCode))
            return

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        # convert the inference result to Numpy array
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr,
                            dtype=np.float32)
        print('result.shape: ', res.shape)

        # postprocess
        prediction = res
        n_labels = len(res)

        img_name = file_name.split('.')[0]
        print("img_name: ", img_name)

        gt = img_gt_dict[img_name]
        print("gt: ", gt)

        sort_index = np.argsort(-prediction)
        print("sort_index: ", sort_index)

        if (n_labels == 1000):
            realLabel = int(gt)
        elif (n_labels == 1001):
            realLabel = int(gt) + 1
        else:
            realLabel = int(gt)
        resCnt = min(len(sort_index), topn)

        with open(os.path.join(res_dir_name, '{}_1.txt'.format(file_name[:-5])), 'w') as f_write:
            res_list = [str(item) + " " for item in sort_index[0: resCnt]]
            print('res_list: ', res_list)
            f_write.writelines(res_list)
            f_write.write('\n')

    # print the total time of inference
    print("The total time of inference is {} s".format(infer_total_time))
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
