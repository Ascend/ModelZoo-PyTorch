"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import argparse
import h5py
import scipy.io
from PIL import Image

parser = argparse.ArgumentParser(description='Transforming arguments')
parser.add_argument('--src', type=str, help='the dir_path include three .mat file ')
parser.add_argument('--dst', type=str, help='the dir_path where to save bounding_box_train/test/query')
arg = parser.parse_args()


def transform_cuhk03(src_root_path, dst_root_path):
    os.makedirs(dst_root_path, exist_ok=True)
    cuhk03 = h5py.File(os.path.join(src_root_path, 'cuhk-03.mat'))
    config = scipy.io.loadmat(os.path.join(
        src_root_path, 'cuhk03_new_protocol_config_detected.mat'))

    train_idx = config['train_idx'].flatten()
    gallery_idx = config['gallery_idx'].flatten()
    query_idx = config['query_idx'].flatten()
    labels = config['labels'].flatten()
    filelist = config['filelist'].flatten()
    cam_id = config['camId'].flatten()

    imgs = cuhk03['detected'][0]
    cam_imgs = []
    for i in range(len(imgs)):
        cam_imgs.append(cuhk03[imgs[i]][:].T)

    def transform_to_path(set_name, idx):
        images_dst_path = os.path.join(dst_root_path, set_name)
        os.makedirs(images_dst_path, exist_ok=True)
        print('transform to', set_name)
        for i in idx:
            i -= 1  # Start from 0
            file_name = filelist[i][0]
            cam_pair_id = int(file_name[0])
            cam_label = int(file_name[2: 5])
            cam_image_idx = int(file_name[8: 10])

            np_image = cuhk03[cam_imgs[cam_pair_id - 1]
            [cam_label - 1][cam_image_idx - 1]][:].T

            unified_cam_id = (cam_pair_id - 1) * 2 + cam_id[i]
            img = Image.fromarray(np_image)

            id_label = str(labels[i]).zfill(4)
            img_dst_path = os.path.join(images_dst_path, id_label)
            os.makedirs(img_dst_path, exist_ok=True)
            print('transform to', img_dst_path)
            img_name = id_label + '_' + 'c' + \
                       str(unified_cam_id) + '_' + str(cam_image_idx).zfill(2)
            img.save(os.path.join(img_dst_path, img_name + '.jpg'))

    transform_to_path('bounding_box_train', train_idx)
    transform_to_path('bounding_box_test', gallery_idx)
    transform_to_path('query', query_idx)


if __name__ == '__main__':
    transform_cuhk03(arg.src, arg.dst)
