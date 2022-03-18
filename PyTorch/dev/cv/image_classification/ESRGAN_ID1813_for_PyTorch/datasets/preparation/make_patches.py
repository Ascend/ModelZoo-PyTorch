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
import cv2
import os
from multiprocessing import Pool
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def crop_image(path, image_size, stride):
    image_list = os.listdir(path)

    for index, image_name in enumerate(image_list):
        if not image_name.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(path, image_name))
        height, width, channels = img.shape
        num_row = height // stride
        num_col = width // stride
        image_index = 0

        if index % 100 == 0:
            print(f'[*] [{index}/{len(image_list)}] Make patch {os.path.join(path, image_name)}')

        for i in range(num_row):
            if (i+1)*image_size > height:
                break
            for j in range(num_col):
                if (j+1)*image_size > width:
                    break
                cv2.imwrite(os.path.join(path, f'{image_name.split(".")[0]}_{image_index}.png'), img[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size])
                image_index += 1
        os.remove(os.path.join(path, image_name))


def make_patches():
    image_dir = [('hr', 128, 100), ('lr', 32, 25)]
    pool = Pool(processes=2)
    pool.starmap(crop_image, image_dir)
    pool.close()
    pool.join()
