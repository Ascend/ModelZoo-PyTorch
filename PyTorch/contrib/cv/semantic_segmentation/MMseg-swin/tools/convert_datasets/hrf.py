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

import mmcv

HRF_LEN = 15
TRAINING_LEN = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert HRF dataset to mmsegmentation format')
    parser.add_argument('healthy_path', help='the path of healthy.zip')
    parser.add_argument(
        'healthy_manualsegm_path', help='the path of healthy_manualsegm.zip')
    parser.add_argument('glaucoma_path', help='the path of glaucoma.zip')
    parser.add_argument(
        'glaucoma_manualsegm_path', help='the path of glaucoma_manualsegm.zip')
    parser.add_argument(
        'diabetic_retinopathy_path',
        help='the path of diabetic_retinopathy.zip')
    parser.add_argument(
        'diabetic_retinopathy_manualsegm_path',
        help='the path of diabetic_retinopathy_manualsegm.zip')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_path = [
        args.healthy_path, args.glaucoma_path, args.diabetic_retinopathy_path
    ]
    annotations_path = [
        args.healthy_manualsegm_path, args.glaucoma_manualsegm_path,
        args.diabetic_retinopathy_manualsegm_path
    ]
    if args.out_dir is None:
        out_dir = osp.join('data', 'HRF')
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

    print('Generating images...')
    for now_path in images_path:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            zip_file = zipfile.ZipFile(now_path)
            zip_file.extractall(tmp_dir)

            assert len(os.listdir(tmp_dir)) == HRF_LEN, \
                'len(os.listdir(tmp_dir)) != {}'.format(HRF_LEN)

            for filename in sorted(os.listdir(tmp_dir))[:TRAINING_LEN]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, 'images', 'training',
                             osp.splitext(filename)[0] + '.png'))
            for filename in sorted(os.listdir(tmp_dir))[TRAINING_LEN:]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, 'images', 'validation',
                             osp.splitext(filename)[0] + '.png'))

    print('Generating annotations...')
    for now_path in annotations_path:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            zip_file = zipfile.ZipFile(now_path)
            zip_file.extractall(tmp_dir)

            assert len(os.listdir(tmp_dir)) == HRF_LEN, \
                'len(os.listdir(tmp_dir)) != {}'.format(HRF_LEN)

            for filename in sorted(os.listdir(tmp_dir))[:TRAINING_LEN]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                # The annotation img should be divided by 128, because some of
                # the annotation imgs are not standard. We should set a
                # threshold to convert the nonstandard annotation imgs. The
                # value divided by 128 is equivalent to '1 if value >= 128
                # else 0'
                mmcv.imwrite(
                    img[:, :, 0] // 128,
                    osp.join(out_dir, 'annotations', 'training',
                             osp.splitext(filename)[0] + '.png'))
            for filename in sorted(os.listdir(tmp_dir))[TRAINING_LEN:]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img[:, :, 0] // 128,
                    osp.join(out_dir, 'annotations', 'validation',
                             osp.splitext(filename)[0] + '.png'))

    print('Done!')


if __name__ == '__main__':
    main()
