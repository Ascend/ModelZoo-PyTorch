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
import glob
import shutil


def rearrange_dir_structure(dataset_path, traintest='train'):
    '''move files to follow the directory structure as REDS

    Original GoPro dataset is organized as GoPro/train/GOPR0854_11_00-000022.png
    We move files and organize them as GoPro/train_GT/GOPR0854_11_00/000022.jpg (similar to REDS).

    :param dataset_path: dataset path
    :return: None
    '''
    os.makedirs(os.path.join(dataset_path, f'{traintest}_GT'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, f'{traintest}_GT_blurred'), exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(f'{dataset_path}/{traintest}', '*')))
    for path in file_list:
        name = os.path.basename(path)
        print(name)

        shutil.move(os.path.join(path, 'sharp'), os.path.join(f'{dataset_path}/{traintest}_GT', name))
        shutil.move(os.path.join(path, 'blur'), os.path.join(f'{dataset_path}/{traintest}_GT_blurred', name))

    shutil.rmtree(os.path.join(dataset_path, traintest))


def generate_meta_info_txt(data_path, meta_info_path):
    '''generate meta_info_GoPro_GT.txt for GoPro

    :param data_path: dataset path.
    :return: None
    '''
    f= open(meta_info_path, "w+")
    file_list = sorted(glob.glob(os.path.join(data_path, 'train_GT/*')))
    total_frames = 0
    for path in file_list:
        name = os.path.basename(path)
        frames = sorted(glob.glob(os.path.join(path, '*')))
        start_frame = os.path.basename(frames[0]).split('.')[0]

        print(name, len(frames), start_frame)
        total_frames += len(frames)

        f.write(f"{name} {len(frames)} (720,1280,3) {start_frame}\r\n")

    assert total_frames == 2103, f'GoPro training set should have 2103 images, but got {total_frames} images'

if __name__ == '__main__':

    dataset_path = 'trainsets/GoPro'

    rearrange_dir_structure(dataset_path, 'train')
    rearrange_dir_structure(dataset_path, 'test')
    generate_meta_info_txt(dataset_path, 'data/meta_info/meta_info_GoPro_GT.txt')


