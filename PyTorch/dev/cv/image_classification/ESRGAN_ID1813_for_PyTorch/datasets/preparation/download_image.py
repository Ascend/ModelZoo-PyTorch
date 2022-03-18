#
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
#
import os
import shutil
from preparation.utils import download_url, unzip_zip_file, unzip_tar_file
from glob import glob
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def download_dataset():
   # DIV2K_HR = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
   # DIV2K_LR = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
   # FLICKR2K = "http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"

   # if not os.path.exists('temp'):
    #    os.makedirs('temp')

    #if not os.path.exists(os.path.join('hr')):
     #   os.makedirs(os.path.join('hr'))
    #if not os.path.exists(os.path.join('lr')):
     #   os.makedirs(os.path.join('lr'))

    #download_url(DIV2K_HR, os.path.join('temp', 'DIV2K_HR.zip'))
    #download_url(DIV2K_LR, os.path.join('temp', 'DIV2K_LR.zip'))
    #download_url(FLICKR2K, os.path.join('temp', 'FLICKR2K.tar'))

    #print('[!] Upzip zipfile')
    #unzip_zip_file(os.path.join('temp', 'DIV2K_HR.zip'), 'temp')
    #unzip_zip_file(os.path.join('temp', 'DIV2K_LR.zip'), 'temp')
    #unzip_tar_file(os.path.join('temp', 'FLICKR2K.tar'), 'temp')

    print('[!] Reformat DIV2K HR')
    image_path = glob('temp/DIV2K_train_HR/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('hr', f'{index:04d}.png'))

    print('[!] Reformat DIV2K LR')
    image_path = glob('temp/DIV2K_train_LR_bicubic/X4/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('lr', f'{index:04d}.png'))

    print('[!] Reformat FLICKR2K HR')
    image_path = glob('temp/Flickr2K/Flickr2K_HR/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('hr', f'{index:05d}.png'))

    print('[!] Reformat FLICKR2K LR')
    image_path = glob('temp/Flickr2K/Flickr2K_LR_bicubic/X4/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('lr', f'{index:05d}.png'))

    shutil.rmtree('temp')
