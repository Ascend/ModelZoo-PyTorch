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
import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import os
import struct
from torch.autograd import Variable
import glob
import sys

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def get_output_path(bin_folder,label_path):
    result_paths = []
    target_paths = []
    print("result_bin_folder:", bin_folder)
    files_source = glob.glob(os.path.join(bin_folder,'*.bin'))
    files_source.sort()
    for file in files_source:
        if file.endswith('.bin'):
            result_path = file
            result_paths.append(result_path)
            name = (result_path.split('/')[3]).split('_')[0]
            target_path = os.path.join(label_path,name+'.bin')
            target_paths.append(target_path)
    return result_paths,target_paths
    
def file2tensor(output_bin,target_bin):
    size = os.path.getsize(output_bin)
    res1 = []
    L = int(size / 4)  
    binfile = open(output_bin, 'rb')
    for i in range(L):
        data = binfile.read(4)
        num = struct.unpack('f', data)
        res1.append(num[0])
    binfile.close()
    dim_res = np.array(res1).reshape(1, 1, 321, 481)  
    tensor_res = torch.tensor(dim_res, dtype=torch.float32)
    
    size = os.path.getsize(target_bin)
    res2 = []
    L = int(size / 4)  
    binfile = open(target_bin, 'rb')
    for i in range(L):
        data = binfile.read(4)
        num = struct.unpack('f', data)
        res2.append(num[0])
    binfile.close()
    dim_res = np.array(res2).reshape(1, 1, 321, 481)  
    tensor_tar = torch.tensor(dim_res, dtype=torch.float32)
    return tensor_res,tensor_tar
          
def post_process(result_path,target_path):
    output_path, target_path= get_output_path(bin_folder=result_path,label_path=label_path)
    psnr_val = 0
    for i in range(len(output_path)):
        output,target = file2tensor(output_path[i],target_path[i])
        Out = torch.clamp(output, 0., 1.)
        psnr = batch_PSNR(Out, target, 1.)
        name = (output_path[i].split('/')[3]).split('_')[0]
        print(name,batch_PSNR(output, target, 1.))
        psnr_val += psnr
    psnr_val /= i
    print('average psnr_val:',psnr_val)

if __name__ == "__main__":
    result_path = sys.argv[1]
    label_path = sys.argv[2]
    post_process(result_path = result_path, target_path = label_path)
    #get_output_path(bin_folder = 'result/dumpOutput_device0')