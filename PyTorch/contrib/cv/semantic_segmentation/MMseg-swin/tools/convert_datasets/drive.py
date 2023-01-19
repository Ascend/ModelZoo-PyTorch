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

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import tempfile
import zipfile

import cv2
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DRIVE dataset to mmsegmentation format')
    parser.add_argument(
        'training_path', help='the training part of DRIVE dataset')
    parser.add_argument(
        'testing_path', help='the testing part of DRIVE dataset')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    training_path = args.training_path
    testing_path = args.testing_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'DRIVE')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print('Extracting training.zip...')
        zip_file = zipfile.ZipFile(training_path)
        zip_file.extractall(tmp_dir)

        print('Generating training dataset...')
        now_dir = osp.join(tmp_dir, 'training', 'images')
        for img_name in os.listdir(now_dir):
            img = mmcv.imread(osp.join(now_dir, img_name))
            mmcv.imwrite(
                img,
                osp.join(
                    out_dir, 'images', 'training',
                    osp.splitext(img_name)[0].replace('_training', '') +
                    '.png'))

        now_dir = osp.join(tmp_dir, 'training', '1st_manual')
        for img_name in os.listdir(now_dir):
            cap = cv2.VideoCapture(osp.join(now_dir, img_name))
            ret, img = cap.read()
            mmcv.imwrite(
                img[:, :, 0] // 128,
                osp.join(out_dir, 'annotations', 'training',
                         osp.splitext(img_name)[0] + '.png'))

        print('Extracting test.zip...')
        zip_file = zipfile.ZipFile(testing_path)
        zip_file.extractall(tmp_dir)

        print('Generating validation dataset...')
        now_dir = osp.join(tmp_dir, 'test', 'images')
        for img_name in os.listdir(now_dir):
            img = mmcv.imread(osp.join(now_dir, img_name))
            mmcv.imwrite(
                img,
                osp.join(
                    out_dir, 'images', 'validation',
                    osp.splitext(img_name)[0].replace('_test', '') + '.png'))

        now_dir = osp.join(tmp_dir, 'test', '1st_manual')
        if osp.exists(now_dir):
            for img_name in os.listdir(now_dir):
                cap = cv2.VideoCapture(osp.join(now_dir, img_name))
                ret, img = cap.read()
                # The annotation img should be divided by 128, because some of
                # the annotation imgs are not standard. We should set a
                # threshold to convert the nonstandard annotation imgs. The
                # value divided by 128 is equivalent to '1 if value >= 128
                # else 0'
                mmcv.imwrite(
                    img[:, :, 0] // 128,
                    osp.join(out_dir, 'annotations', 'validation',
                             osp.splitext(img_name)[0] + '.png'))

        now_dir = osp.join(tmp_dir, 'test', '2nd_manual')
        if osp.exists(now_dir):
            for img_name in os.listdir(now_dir):
                cap = cv2.VideoCapture(osp.join(now_dir, img_name))
                ret, img = cap.read()
                mmcv.imwrite(
                    img[:, :, 0] // 128,
                    osp.join(out_dir, 'annotations', 'validation',
                             osp.splitext(img_name)[0] + '.png'))

        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
