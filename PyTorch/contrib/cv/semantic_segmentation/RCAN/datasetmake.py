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

import os 
from os.path import join
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import zipfile
import shutil

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def data_augmentation(paths_input,paths_output):
    idx = 0
    for path in tqdm(os.listdir(paths_input)):
        path = os.path.join(paths_input , path)
        if is_image_file(path) == False:
            continue
        img = cv2.imread(path)
        x = 0
        y = 0
        while y < img.shape[0]:
            if x+96 < img.shape[1] and y+96 < img.shape[0]:
                img_org = img[y:y+96 , x:x+96]
                filename = paths_output + str(idx) + '.png'
                cv2.imwrite(filename , img_org)
                idx = idx+1
                # img_flip = cv2.flip(img_org , 0)
                # filename = paths_output + str(idx) + '.png'
                # cv2.imwrite(filename , img_flip)
                # idx = idx+1
                # img_rotate_90 = rotate_image(img_org , 90)
                # filename = paths_output + str(idx) + '.png'
                # cv2.imwrite(filename , img_rotate_90)
                # idx = idx+1
                # img_rotate_90_flip = cv2.flip(img_rotate_90 , 0)
                # filename = paths_output + str(idx) + '.png'
                # cv2.imwrite(filename , img_rotate_90_flip)
                # idx = idx+1
                # img_rotate_180 = rotate_image(img_org , 180)
                # filename = paths_output + str(idx) + '.png'
                # cv2.imwrite(filename , img_rotate_180)
                # idx = idx+1
                # img_rotate_180_flip = cv2.flip(img_rotate_180 , 0)
                # filename = paths_output + str(idx) + '.png'
                # cv2.imwrite(filename , img_rotate_180_flip)
                # idx = idx+1
                # img_rotate_270 = rotate_image(img_org , 270)
                # filename = paths_output + str(idx) + '.png'
                # cv2.imwrite(filename , img_rotate_270)
                # idx = idx+1
                # img_rotate_270_flip = cv2.flip(img_rotate_270 , 0)
                # filename = paths_output + str(idx) + '.png'
                # cv2.imwrite(filename , img_rotate_270_flip)
                # idx = idx+1
                x = x + 96
            else:
                x = 0
                y = y + 96
            # idx = idx + 1
        # break
    print(idx)

def un_zip(file_name,unzip_path):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(unzip_path):
        pass
    else:
        os.mkdir(unzip_path)
    print(file_name,"unziping...")
    for names in zip_file.namelist():
        # print(names)
        zip_file.extract(names, unzip_path)
    zip_file.close()
    print(file_name,"ended unzip for {} files and {} dir".format(len(zip_file.namelist())-1 , 1))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_zip_path', type=str,default = "/root/dataset_zzq/", help = "The path of two zip files")
    parser.add_argument('--dataset_path', type=str, default="/root/dataset_RCAN/",help = "The path of final dataset")
    opt = parser.parse_args()

    input_zip_path1 = os.path.join(opt.input_zip_path,"DIV2K_train_HR.zip")
    input_zip_path2 = os.path.join(opt.input_zip_path,"DIV2K_valid_HR.zip")
    if not os.path.exists(opt.dataset_path):
        os.makedirs(opt.dataset_path)
    un_zip(input_zip_path1,opt.dataset_path)
    un_zip(input_zip_path2,opt.dataset_path)

    file_list1 = os.listdir(os.path.join(opt.dataset_path,"DIV2K_train_HR"))
    file_list2 = os.listdir(os.path.join(opt.dataset_path,"DIV2K_valid_HR"))
    if not os.path.exists(os.path.join(opt.dataset_path,"DIV2K_HR")):
        os.makedirs(os.path.join(opt.dataset_path,"DIV2K_HR"))
    for file_path in file_list1:
        shutil.move(os.path.join(opt.dataset_path,"DIV2K_train_HR",file_path),os.path.join(opt.dataset_path,"DIV2K_HR",file_path))
    for file_path in file_list2:
        shutil.move(os.path.join(opt.dataset_path,"DIV2K_valid_HR",file_path),os.path.join(opt.dataset_path,"DIV2K_HR",file_path))
    shutil.rmtree(os.path.join(opt.dataset_path,"DIV2K_train_HR"))
    shutil.rmtree(os.path.join(opt.dataset_path,"DIV2K_valid_HR"))

    data_augmentation(os.path.join(opt.dataset_path,"DIV2K_HR"),opt.dataset_path)
    shutil.rmtree(os.path.join(opt.dataset_path,"DIV2K_HR"))
    

