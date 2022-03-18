# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
import argparse
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def make_power(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)
    
    
def preprocess(PIL_img, image_shape=256):
    process=transforms.Compose([
        transforms.Lambda(lambda img: make_power(img, base=4, method=Image.BICUBIC)),
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
    return process(PIL_img)
    
    
def postprocess(img_tensor):
    inv_normalize = transforms.Normalize(
        mean= (-1,-1,-1),
        std= (2.0,2.0,2.0))
    to_PIL_image = transforms.ToPILImage().convert('RGB')
    return to_PIL_image(inv_normalize(img_tensor[0]).clamp(0, 1))
    
    
def parse():
    """Define the common options that are used in both training and test."""
    # basic parameters
    parser = argparse.ArgumentParser(description='cyclegan test for image preprocess')
    parser.add_argument('--src_path_testA', required=False,default='datasets/maps/testA/', help='path to images testA)')
    parser.add_argument('--save_path_testA_bin', type=str, default='nputest/testa', help='name of the ..')
    parser.add_argument('--path_testA_binName', type=str, default='testA_prep_bin.info', help='name of the ..')
    parser.add_argument('--src_path_testB', required=False, default='datasets/maps/testB/', help='path to images testB)')
    parser.add_argument('--save_path_testB_bin', type=str, default='nputest/testb', help='name of the ..')
    parser.add_argument('--path_testB_binName', type=str, default='testB_prep_bin.info', help='name of the ..')
    opt=parser.parse_args()
    if(os.path.exists(opt.save_path_testA_bin)==False):
        os.makedirs(opt.save_path_testA_bin)
    if(os.path.exists(opt.save_path_testB_bin)==False):
        os.makedirs(opt.save_path_testB_bin)
    return opt
    
    
def rs_img_bin(src_path,save_path,data_list_path):
    i = 0
    in_files = os.listdir(src_path)
    listfile = open(data_list_path, 'w')
    for file in in_files:
        #print(file, "===", i)
        input_image = Image.open(src_path + '/' + file)
        input_tensor = preprocess(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))
        temp = str(str(i) + ' ./' + os.path.join(save_path, file.split('.')[0] + ".bin") + ' ' + '256 256\n')
        listfile.write(temp)
        i = i + 1
    listfile.close()
    
    
def main(opt):
    # deal testA and save img data to bin
    rs_img_bin(opt.src_path_testA, opt.save_path_testA_bin, opt.path_testA_binName)
    # deal testB and save img data to bin
    rs_img_bin(opt.src_path_testB, opt.save_path_testB_bin, opt.path_testB_binName)
    return 0
    
    
if __name__=='__main__':
    opt=parse()
    main(opt)