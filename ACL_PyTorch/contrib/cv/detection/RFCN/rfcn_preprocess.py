# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('./RFCN-pytorch.1.0')
import numpy as np
import cv2
from imageio import imread
from lib.model.utils.config import cfg
import os
import torch
import argparse
from past.builtins import xrange
from torch.autograd import Variable
import glob
import torch.nn.functional as F

def im_list_to_blob(ims):
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    for target_size in cfg.TEST.SCALES:
        im = im.astype(np.float32, copy=False)
        im -= cfg.PIXEL_MEANS
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def get_jpg_info(file_path, output_path,img_name):
    processed_ims = []
    im_scales = []
    im_file = os.path.join(file_path,img_name+".jpg")
    print(im_file)
    im = imread(im_file)
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.concatenate((im, im, im), axis=2)
    # rgb -> bgr
    im = im[:,:,::-1]
    im, im_scale = _get_image_blob(im)
    im_scales.append(im_scale)
    processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, im_scales


def get_bin(file_path,bin_path):
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    with open("./RFCN-pytorch.1.0/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt") as f:
        imglist = [x.strip() for x in f.readlines()]
    num_images = len(imglist)
    for i in range(num_images):
        im_data = torch.Tensor([0.])
        im_blob, im_scales = get_jpg_info(file_path,bin_path,imglist[i])
        data = torch.from_numpy(im_blob)
        data = data.permute(0, 3, 1, 2)
        pad_value = 0
        batch_shape = (3, 1344, 1344)
        padding_size = [0, batch_shape[-1] - data[0].shape[-1],
                        0, batch_shape[-2] - data[0].shape[-2]]
        data = F.pad(data, padding_size, value=pad_value)
        im_data.resize_(data.size()).copy_(data)
        im_data1 = im_data.numpy()
        im_data1.tofile(os.path.join(bin_path, imglist[i].split('.')[0] + ".bin"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess of RFCN pytorch model')
    parser.add_argument("--file_path", dest="file_path",default="./RFCN-pytorch.1.0/data/VOCdevkit2007/VOC2007/JPEGImages/", help='image of dataset')
    parser.add_argument("--bin_path",dest="bin_path", default="./bin", help='Preprocessed image buffer')

    args = parser.parse_args()

    file_path = args.file_path
    bin_path = args.bin_path
    get_bin(file_path,bin_path)
    



