# Copyright 2020 Huawei Technologies Co., Ltd
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
import sys
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import struct
from skimage.measure.simple_metrics import compare_psnr


def batch_PSNR(img, imclean, data_range):

    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def bin2npy(filepath):

    size = os.path.getsize(filepath)  
    res = []
    L = int(size / 4)
    binfile = open(filepath, 'rb')  
    for i in range(L):
        data = binfile.read(4)  
        num = struct.unpack('f', data)
        res.append(num[0])
    binfile.close()
    dim_res = np.array(res).reshape(1, 1, 481, 481)
    return dim_res


def main(Result_path):

    # load data info
    print('Loading ISource bin ...\n')
    ISource = glob.glob(os.path.join('ISource', '*.bin'))
    ISource.sort()
    print('Loading INoisy bin ...\n')
    INoisy = glob.glob(os.path.join('INoisy', '*.bin'))
    INoisy.sort()
    # load result file
    print('Loading res bin ...\n')
    Result_path = glob.glob(os.path.join(Result_path, '*.bin'))
    Result_path.sort()

    # begin data
    print('begin infer')
    psnr_test = 0
    n_lables = 0

    for isource in ISource:
        isource_name = isource
        # isource
        isource = bin2npy(isource)
        isource = torch.from_numpy(isource)
        # inoisy
        inoisy = bin2npy(INoisy[n_lables])
        inoisy = torch.from_numpy(inoisy)
        # Result_path
        Result = bin2npy(Result_path[n_lables])
        Result = torch.from_numpy(Result)
        n_lables += 1
        print('infering...')
        with torch.no_grad(): 
            Out = torch.clamp(inoisy - Result, 0., 1.)
        psnr = batch_PSNR(Out, isource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (isource_name, psnr))
    psnr_test /= len(ISource)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    
    try:
        Result_path = sys.argv[1]

    except IndexError:
        print("Stopped!")
        exit(1)

    if not (os.path.exists(Result_path)):
        print("Result path doesn't exist.")

    main(Result_path)
