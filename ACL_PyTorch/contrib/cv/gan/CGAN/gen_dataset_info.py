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
import glob
import argparse

def parse_args():
    desc = "Pytorch implementation of CGAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_bin', type=str, default='./prep_dataset', help="The input_dim")
    parser.add_argument('--info_name', type=str, default='CGAN_prep_bin.info', help="The output_dim")
    parser.add_argument('--width', type=str, default='78', help="The width of input ")
    parser.add_argument('--height', type=str, default='100', help="The height of input")
    return parser.parse_args()
    
def get_bin_info(img_root_path='./data', info_name='CGAN_prep_bin.info', width='72', height='100'):
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
    args = parse_args()
    get_bin_info(img_root_path=args.dataset_bin, info_name=args.info_name, width=args.width, height=args.height)
