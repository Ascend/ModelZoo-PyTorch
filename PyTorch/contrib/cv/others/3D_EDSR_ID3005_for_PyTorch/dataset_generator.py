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
# cropping Train/Test image patches
import os
from skimage import io
from option import args

# Train and Test image save paths
Train_LR_folder = os.path.join(args.dir_train_data, 'LR')
Train_HR_folder = os.path.join(args.dir_train_data, 'HR')
Test_LR_folder = os.path.join(args.dir_test_data, 'LR')
Test_HR_folder = os.path.join(args.dir_test_data, 'HR')
if not os.path.exists(Train_LR_folder):
    os.makedirs(Train_LR_folder)
if not os.path.exists(Train_HR_folder):
    os.makedirs(Train_HR_folder)
if not os.path.exists(Test_LR_folder):
    os.makedirs(Test_LR_folder)
if not os.path.exists(Test_HR_folder):
    os.makedirs(Test_HR_folder)
# generate LR data
Core1_Subvol1_LR = io.imread(os.path.join(args.dir_data, 'Core1_Subvol1_LR.tif'))
Image_Size = 210
LR_Image_Size = 15
HR_Image_Size = LR_Image_Size * 3
Train_Dataset_Size = 801
Test_Dataset_Size = 10
count1 = 1
for z in range(LR_Image_Size, Image_Size, LR_Image_Size):
    for y in range(LR_Image_Size, Image_Size, LR_Image_Size):
        for x in range(LR_Image_Size, Image_Size, LR_Image_Size):
            img = Core1_Subvol1_LR[z - LR_Image_Size:z + LR_Image_Size, y - LR_Image_Size:y + LR_Image_Size, x - LR_Image_Size:x + LR_Image_Size]
            if count1 < Train_Dataset_Size:
                io.imsave(os.path.join(Train_LR_folder, str(count1) + '.tif'), img)
                count1 += 1
            elif count1 < Train_Dataset_Size + Test_Dataset_Size:
                io.imsave(os.path.join(Test_LR_folder, str(count1) + '.tif'), img)
                count1 += 1
            else:
                break

# generate HR data
Core1_Subvol1_HR = io.imread(os.path.join(args.dir_data, 'Core1_Subvol1_HR.tif'))
count2 = 1
for z in range(HR_Image_Size, Image_Size * 3, HR_Image_Size):
    for y in range(HR_Image_Size, Image_Size * 3, HR_Image_Size):
        for x in range(HR_Image_Size, Image_Size * 3, HR_Image_Size):
            img = Core1_Subvol1_HR[z - HR_Image_Size:z + HR_Image_Size, y - HR_Image_Size:y + HR_Image_Size, x - HR_Image_Size:x + HR_Image_Size]
            if count2 < Train_Dataset_Size:
                io.imsave(os.path.join(Train_HR_folder, str(count2) + '.tif'), img)
                count2 += 1
            elif count2 < Train_Dataset_Size + Test_Dataset_Size:
                io.imsave(os.path.join(Test_HR_folder, str(count2) + '.tif'), img)
                count2 += 1
            else:
                break
