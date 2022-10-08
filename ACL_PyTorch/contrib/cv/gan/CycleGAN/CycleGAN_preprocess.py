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
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

def parse():
    """Define the common options that are used in both training and test."""
    # basic parameters
    parser = argparse.ArgumentParser(description='cyclegan test for image preprocess')
    parser.add_argument('--src_path_testA', required=False, default='datasets/maps/testA/',
                        help='path to images testA)')
    parser.add_argument('--save_pathTestA_dst', required=False, default='datasetsDst/maps/testA/',
                        help='path to images testA)')
    parser.add_argument('--src_path_testB', required=False, default='datasets/maps/testB/',
                        help='path to images testB)')
    parser.add_argument('--save_pathTestB_dst', required=False, default='datasetsDst/maps/testB/',
                        help='path to images testA)')
    opt = parser.parse_args()
    if (os.path.exists(opt.save_pathTestA_dst) == False):
        os.makedirs(opt.save_pathTestA_dst)
    if (os.path.exists(opt.save_pathTestB_dst) == False):
        os.makedirs(opt.save_pathTestB_dst)
    return opt

def make_power(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)


def preprocess(PIL_img, image_shape):
    process = transforms.Compose([
        transforms.Lambda(lambda img: make_power(img, base=4, method=Image.BICUBIC)),
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return process(PIL_img).unsqueeze(dim=0)  # (batch_size, 3, H, W)


def rs_img_bin(src_path, savepath):
    in_files = os.listdir(src_path)
    for file in tqdm(in_files):
        image_path = src_path + '/' + file
        input_image = Image.open(image_path).convert('RGB')
        tensorData = preprocess(input_image, 256)
        image = np.array(tensorData).astype(np.float32)
        image.tofile(os.path.join(savepath, str(file).split('.')[0] + ".bin"))



def main(opt):
    # deal testA and save img data to bin
    rs_img_bin(opt.src_path_testA, opt.save_pathTestA_dst)
    # deal testB and save img data to bin
    rs_img_bin(opt.src_path_testB, opt.save_pathTestB_dst)
    return 0


if __name__ == '__main__':
    opt = parse()
    main(opt)
