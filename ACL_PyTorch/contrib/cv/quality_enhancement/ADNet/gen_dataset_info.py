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
import argparse
import glob
import sys

def get_bin_info(img_root_path='./prep_dataset/INoisy',
                 info_name='ADNet_prep_bin.info', width='481', height='321'):
    img_path = []
    files_source = glob.glob(os.path.join(img_root_path,'*.bin'))
    files_source.sort()
    for file in files_source:
        if file.endswith('.bin'):
            imgpath = file
            img_path.append(imgpath)
    with open(info_name, 'w') as fp:
        for index in range(len(img_path)):
            content = ' '.join([str(index), img_path[index], width, height])
            fp.write(content)
            fp.write('\n')

if __name__ == '__main__':
    dataset_bin = sys.argv[1]
    info_name = sys.argv[2]
    width = sys.argv[3]
    height = sys.argv[4]
    get_bin_info(img_root_path=dataset_bin, info_name=info_name, width=width, height=height)
