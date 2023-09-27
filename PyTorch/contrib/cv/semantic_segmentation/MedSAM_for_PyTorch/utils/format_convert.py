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
import numpy as np
from skimage import io
import SimpleITK as sitk


def dcm2nii(dcm_path, nii_path):
    """
    Convert dicom files to nii files
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, nii_path)

def mhd2nii(mhd_path, nii_path):
    """
    Convert mhd files to nii files
    """
    image = sitk.ReadImage(mhd_path)
    sitk.WriteImage(image, nii_path)

def nii2nii(nii_path, nii_gz_path):
    """
    Convert nii files to nii.gz files, which can reduce the file size
    """
    image = sitk.ReadImage(nii_path)
    sitk.WriteImage(image, nii_gz_path)

def nrrd2nii(nrrd_path, nii_path):
    """
    Convert nrrd files to nii files
    """
    image = sitk.ReadImage(nrrd_path)
    sitk.WriteImage(image, nii_path)

def jpg2png(jpg_path, png_path):
    """
    Convert jpg files to png files
    """
    image = io.imread(jpg_path)
    io.imsave(png_path, image)

def patchfy(img, mask, outpath, basename):
    """
    Patchfy the image and mask into 1024x1024 patches
    """
    image_patch_dir = join(outpath, "images")
    mask_patch_dir = join(outpath, "labels")
    os.makedirs(image_patch_dir, exist_ok=True)
    os.makedirs(mask_patch_dir, exist_ok=True)
    assert img.shape[:2] == mask.shape
    patch_height = 1024
    patch_width = 1024

    img_height, img_width = img.shape[:2]
    mask_height, mask_width = mask.shape

    if img_height % patch_height != 0:
        img = np.pad(img, ((0, patch_height - img_height % patch_height), (0, 0), (0, 0)), mode="constant")
    if img_width % patch_width != 0:
        img = np.pad(img, ((0, 0), (0, patch_width - img_width % patch_width), (0, 0)), mode="constant")
    if mask_height % patch_height != 0:
        mask = np.pad(mask, ((0, patch_height - mask_height % patch_height), (0, 0)), mode="constant")
    if mask_width % patch_width != 0:
        mask = np.pad(mask, ((0, 0), (0, patch_width - mask_width % patch_width)), mode="constant")

    assert img.shape[:2] == mask.shape
    assert img.shape[0] % patch_height == 0
    assert img.shape[1] % patch_width == 0
    assert mask.shape[0] % patch_height == 0
    assert mask.shape[1] % patch_width == 0

    height_steps = (img_height // patch_height) if img_height % patch_height == 0 else (img_height // patch_height + 1)
    width_steps = (img_width // patch_width) if img_width % patch_width == 0 else (img_width // patch_width + 1)

    for i in range(height_steps):
        for j in range(width_steps):
            img_patch = img[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :]
            mask_patch = mask[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
            assert img_patch.shape[:2] == mask_patch.shape
            assert img_patch.shape[0] == patch_height
            assert img_patch.shape[1] == patch_width
            print(f"img_patch.shape: {img_patch.shape}, mask_patch.shape: {mask_patch.shape}")
            img_patch_path = join(image_patch_dir, f"{basename}_{i}_{j}.png")
            mask_patch_path = join(mask_patch_dir, f"{basename}_{i}_{j}.png")
            io.imsave(img_patch_path, img_patch)    
            io.imsave(mask_patch_path, mask_patch)


def rle_decode(mask_rle, img_shape):
    """
    #functions to convert encoding to mask and mask to encoding
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths
    img = np.zeros((np.product(img_shape),), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = 255
    # https://stackoverflow.com/a/46574906/4521646
    img.shape = img_shape
    return img.T