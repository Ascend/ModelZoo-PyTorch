'''
The scripts to execute sdk infer
'''
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
import time
import numpy as np
import cv2
from glob import glob
from metrics import iou_score

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="ENET process")
    parser.add_argument("--pipeline", type=str,
                        default='../data/config/unet.pipeline', help="SDK infer pipeline")
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')

     # dataset
    parser.add_argument('--dataset', default='inputs/dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
                        

    parser.add_argument('--save_mask', default=1, type=int,
                        help='0 for False, 1 for True')
    parser.add_argument('--mask_result_path', default='./mask_result', type=str,
                        help='the folder to save the semantic mask images')
    args_opt = parser.parse_args()
    return args_opt


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

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

def normalize_cv2(img, mean, denominator):
    dim = 4
    if mean.shape and len(mean) != dim and mean.shape != img.shape:
        mean = np.array(mean.tolist() + [0] * (dim - len(mean)), dtype=np.float64)
    if not denominator.shape:
        denominator = np.array([denominator.tolist()] * dim, dtype=np.float64)
    elif len(denominator) != dim and denominator.shape != img.shape:
        denominator = np.array(denominator.tolist() + [1] * (dim - len(denominator)), dtype=np.float64)

    img = np.ascontiguousarray(img.astype("float32"))
    cv2.subtract(img, mean.astype(np.float64), img)
    cv2.multiply(img, denominator.astype(np.float64), img)
    return img

def normalize_numpy(img, mean, denominator):
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def normalize(img):
    max_pixel_value=255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    mean *= max_pixel_value

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    
    return normalize_cv2(img, mean, denominator)

def _get_image(img_id, class_num, args):

    img_dir=os.path.join(args.dataset, 'images')
    mask_dir=os.path.join(args.dataset, 'masks')
    
    img = cv2.imread(os.path.join(img_dir, img_id + args.img_ext))
    mask = []
    for i in range(class_num):
        mask.append(cv2.imread(os.path.join(mask_dir, str(i), img_id + args.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
    mask = np.dstack(mask)
    img = normalize(img)
    img = img.astype('float32') / 255 
    mask = mask.astype('float32') / 255
    img = img.transpose(2, 0, 1)  
    mask = mask.transpose(2, 0, 1)
    return img, mask, {'img_id': img_id}


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self, name, fmt=':f', start_count_index=0):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # iou.update(iou_now, input.size(0))
    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__) 

def main():
    """
    read pipeline and do infer
    """
    class_num = 1
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

    stream_name = b'unet'
    infer_total_time = 0
    assert os.path.exists(args.dataset), "Please put dataset in " + str(args.dataset)
    
    
    # Data loading code
    img_ids = glob(os.path.join(args.dataset, 'images', '*' + args.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    
    if not img_ids:
        raise RuntimeError(
            "Found 0 images in subfolders of:" + args.dataset + "\n")

    for c in range(class_num):
        os.makedirs(os.path.join(args.mask_result_path, str(c)), exist_ok=True)        

    batch_time = AverageMeter('Time', ':6.3f')
    iou = AverageMeter('Iou', ':6.4f')
    end = time.time()
    for img_id in img_ids:
        print("Processing ---> ", img_id)
        #img_ids, base_dir, img_ext, mask_ext, idx, class_num
        img, mask, _ = _get_image(img_id, class_num, args)
        img = np.expand_dims(img, 0)  # NCHW image shape: (1,3,H,W) [0,1]
        mask = np.expand_dims(mask, 0)  # NHW
        #to do inference 
        if not send_source_data(0, img, stream_name, stream_manager_api):
            return
        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time


        #check infer result
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
            result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    
        mask_image = res.reshape(1, 1, 96, 96)
        iou_now = iou_score(mask_image, mask)
        iou.update(iou_now, img.shape[0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        mask_image = sigmoid(mask_image)  #值映射到0-1
        for i in range(len(mask_image)):
                for c in range(class_num):
                    cv2.imwrite(os.path.join(args.mask_result_path, str(c), img_id + '.png'),
                                (mask_image[i, c] * 255).astype('uint8'))

    print('[AVG-IOU] * Iou {iou.avg:.4f}'
                  .format(iou=iou))
 
    print("Testing finished....")
    print("=======================================")
    print("The total time of inference is {} s".format(infer_total_time))
    print("=======================================")

    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()
