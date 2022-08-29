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


def w2txt(file_path, filename_list, data):
    with open(file_path, "w") as file:
        for i in range(data.shape[0]):
            s = filename_list[i].split('.')[0] + ' '
            s += ' '.join(str(format(num, '.7f')) for num in data[i])
            file.write(s+"\n")


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
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    img_gt_dict = {}
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            imgName = temp[0].split(".")[0]
            imgLab = temp[1]
            img_gt_dict[imgName] = imgLab
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
    with open('../data/config/spansnet_100.pipline', 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'spansnet_100'

    # Construct the input of the stream
    # data_input = MxDataInput()
    infer_total_time = 0
    # image input
    image_input_dir = sys.argv[1]
    # label input
    real_label_name = sys.argv[2]
    #result
    json_file_name = sys.argv[3]
    file_list = os.listdir(image_input_dir)  #

    writer = open(json_file_name, 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []

    input_size = (256, 256)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_gt_dict = cre_groundtruth_dict_fromtxt(real_label_name)
    topn = 5
    count_hit = np.zeros(topn)
    count = 0
    resCnt = 0
    n_labels = 0
    res_sents = []
    filename_list = []
    for file in file_list:
        print(file, "====", count)
        count += 1
        #preprocess
        img = Image.open(os.path.join(image_input_dir, file)).convert('RGB')
        img = resize(img, input_size)  # transforms.Resize(256)
        img = np.array(img, dtype=np.float32)
        img = center_crop(img, 224, 224)   # transforms.CenterCrop(224)
        img = img / 255.  # transforms.ToTensor()
        # mean and variance
        img[..., 0] = (img[..., 0] - mean[0]) / std[0]
        img[..., 1] = (img[..., 1] - mean[1]) / std[1]
        img[..., 2] = (img[..., 2] - mean[2]) / std[2]
        img = img.transpose(2, 0, 1) # HWC -> CHW
        tensor = img.reshape(1, 3, 224, 224)
        # infer
        if not send_source_data(0, tensor, stream_name, stream_manager_api):
            return

        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time

        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
        res_sents.append(res)
        filename_list.append(file.split('.')[0])

        # postprocess
        prediction = res
        n_labels = len(res)

        img_name = file.split('.')[0]
        gt = img_gt_dict[img_name]
        sort_index = np.argsort(-prediction)
        if (n_labels == 1000):
            realLabel = int(gt)
        elif (n_labels == 1001):
            realLabel = int(gt) + 1
        else:
            realLabel = int(gt)
        resCnt = min(len(sort_index), topn)
        for i in range(resCnt):
            if (str(realLabel) == str(sort_index[i])):
                count_hit[i] += 1
                break

    file_path = "sdk_prediction_result.txt"
    res_sents = np.array(res_sents).astype(np.float32)
    w2txt(file_path, filename_list, res_sents)

    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(count)},
             {"key": "Number of classes", "value": str(n_labels)}])
        if count == 0:
            accuracy = 0
        else:
            accuracy = np.cumsum(count_hit) / count
        for i in range(resCnt):
            table_dict["value"].append({"key": "Top" + str(i + 1) + " accuracy",
                                        "value": str(
                                            round(accuracy[i] * 100, 2)) + '%'})
        json.dump(table_dict, writer)
    writer.close()
    # print the total time of inference
    print("The total time of inference is {} s".format(infer_total_time))

    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
