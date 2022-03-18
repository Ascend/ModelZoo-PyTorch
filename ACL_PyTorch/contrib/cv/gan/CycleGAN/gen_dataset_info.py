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


def parse():
    """Define the common options that are used in both training and test."""
    # basic parameters
    parser = argparse.ArgumentParser(description='cyclegan test for image preprocess')
    parser.add_argument('--src_path_testA', required=False, default='datasets/maps/testA/',
                        help='path to images testA)')
    parser.add_argument('--save_pathTestA_dst', required=False, default='datasetsDst/maps/testA/',
                        help='path to images testA)')
    parser.add_argument('--dataTestA_infoName', default='testA_prep.info', help='name of the ..')

    parser.add_argument('--src_path_testB', required=False, default='datasets/maps/testB/',
                        help='path to images testB)')
    parser.add_argument('--save_pathTestB_dst', required=False, default='datasetsDst/maps/testB/',
                        help='path to images testA)')
    parser.add_argument('--dataTestB_infoName', required=False, default='testB_prep.info', help='name of the ..')
    opt = parser.parse_args()
    if (os.path.exists(opt.save_pathTestA_dst) == False):
        os.makedirs(opt.save_pathTestA_dst)
    if (os.path.exists(opt.save_pathTestB_dst) == False):
        os.makedirs(opt.save_pathTestB_dst)
    return opt


def rs_img_bin(src_path, savepath, data_list_path):
    i = 0
    in_files = os.listdir(src_path)
    listfile = open(data_list_path, 'w')
    for file in in_files:
        # print(file, "===", i)
        image_path = src_path + '/' + file
        input_image = Image.open(image_path)
        imgsavepath = savepath + str(file).split('.')[0] + '.jpeg'
        input_image.thumbnail((512, 512), Image.ANTIALIAS)
        input_image.save(imgsavepath)
        w, h = input_image.size
        temp = str(i) + ' ' + savepath + '/' + str(file).split('.')[0] + \
               '.jpeg' + ' ' + str(w) + ' ' + str(h) + '\n'
        listfile.write(temp)
        i = i + 1
    listfile.close()


def main(opt):
    # deal testA and save img data to bin
    rs_img_bin(opt.src_path_testA, opt.save_pathTestA_dst, opt.dataTestA_infoName)
    # deal testB and save img data to bin
    rs_img_bin(opt.src_path_testB, opt.save_pathTestB_dst, opt.dataTestB_infoName)
    return 0


if __name__ == '__main__':
    opt = parse()
    main(opt)
