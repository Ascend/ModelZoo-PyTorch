# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

import os
join = os.path.join
import random

path_nii = '' # please complete path; two subfolders: images and labels
path_video = None # or specify the path
path_2d = None # or specify the path

#%% split 3D nii data
if path_nii is not None:
    img_path = join(path_nii, 'images')
    gt_path = join(path_nii, 'labels')
    gt_names = sorted(os.listdir(gt_path))
    img_suffix = '_0000.nii.gz'
    gt_suffix = '.nii.gz'
    # split 20% data for validation and testing
    validation_path = join(path_nii, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    testing_path = join(path_nii, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)
    candidates = random.sample(gt_names, int(len(gt_names)*0.2))
    # split half of test names for validation
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]
    # move validation and testing data to corresponding folders
    for name in validation_names:
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(validation_path, 'images', img_name))
        os.rename(join(gt_path, name), join(validation_path, 'labels', name))
    for name in test_names:
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(testing_path, 'images', img_name))
        os.rename(join(gt_path, name), join(testing_path, 'labels', name))


##% split 2D images
if path_2d is not None:
    img_path = join(path_2d, 'images')
    gt_path = join(path_2d, 'labels')
    gt_names = sorted(os.listdir(gt_path))
    img_suffix = '.png'
    gt_suffix = '.png'
    # split 20% data for validation and testing
    validation_path = join(path_2d, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    testing_path = join(path_2d, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)
    candidates = random.sample(gt_names, int(len(gt_names)*0.2))
    # split half of test names for validation
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]
    # move validation and testing data to corresponding folders
    for name in validation_names:
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(validation_path, 'images', img_name))
        os.rename(join(gt_path, name), join(validation_path, 'labels', name))

    for name in test_names:
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(testing_path, 'images', img_name))
        os.rename(join(gt_path, name), join(testing_path, 'labels', name))

#%% split video data
if path_video is not None:
    img_path = join(path_video, 'images')
    gt_path = join(path_video, 'labels')
    gt_folders = sorted(os.listdir(gt_path))
    # split 20% videos for validation and testing
    validation_path = join(path_video, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    testing_path = join(path_video, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)
    candidates = random.sample(gt_folders, int(len(gt_folders)*0.2))
    # split half of test names for validation
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]
    # move validation and testing data to corresponding folders
    for name in validation_names:
        os.rename(join(img_path, name), join(validation_path, 'images', name))
        os.rename(join(gt_path, name), join(validation_path, 'labels', name))
    for name in test_names:
        os.rename(join(img_path, name), join(testing_path, 'images', name))
        os.rename(join(gt_path, name), join(testing_path, 'labels', name))
