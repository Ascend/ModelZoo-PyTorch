# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os


def get_bin_info(img_root_path='prep_dataset/datasets/leftImg8bit',
                 mask_folder='opt/npu/prep_dataset/datasets/cityscapes/gtFine',
                 info_name='fast_scnn_prep_bin.info', width='2048', height='1024'):
    img_path = []
    mask_path = []
    for root, _, files in os.walk(img_root_path):
        for filename in files:
            if filename.startswith('._'):
                continue
            if filename.endswith('.bin'):
                imgpath = os.path.join(root, filename)
                img_path.append(imgpath)
                foldername = os.path.basename(os.path.dirname(imgpath))
                maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                maskpath = os.path.join(mask_folder, foldername, maskname)
                mask_path.append(maskpath)

    with open(info_name, 'w') as fp:
        for index in range(len(img_path)):
            content = ' '.join([str(index), img_path[index], width, height])
            fp.write(content)
            fp.write('\n')


if __name__ == '__main__':
    img_root_path = 'prep_dataset/datasets/leftImg8bit/'
    mask_folder = '/opt/npu/prep_dataset/datasets/gtFine'
    info_name = 'fast_scnn_prep_bin.info'
    width = '2048'
    height = '1024'
    get_bin_info(img_root_path=img_root_path, mask_folder=mask_folder, info_name=info_name, width=width, height=height)
