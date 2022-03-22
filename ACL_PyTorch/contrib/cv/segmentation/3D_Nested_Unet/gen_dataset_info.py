"""
    Copyright 2020 Huawei Technologies Co., Ltd

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    Typical usage example:
"""
import os
import sys
from glob import glob
import pdb


def get_bin_info(file_path, info_name, shape, split4=True):
    """
    @description: get given bin information
    @param file_path  bin file path
    @param info_name given information name
    @param shape  image shape
    @return
    """
    bin_images = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, shape[0], shape[1]])
            file.write(content)
            file.write('\n')
    print('共计.bin文件个数：', len(bin_images))
    print('info已写入：', os.path.abspath(info_name))
    if split4:  # 是否切割为4卡的info
        sths = ['sth1.info', 'sth2.info', 'sth3.info', 'sth4.info']
        length = len(bin_images)
        step = length // 4
        b1 = bin_images[0: step]
        b2 = bin_images[step: 2*step]
        b3 = bin_images[2*step: 3*step]
        b4 = bin_images[3*step:]
        with open(sths[0], 'w') as file:
            for index, img in enumerate(b1):
                content = ' '.join([str(index), img, shape[0], shape[1]])
                file.write(content)
                file.write('\n')
        with open(sths[1], 'w') as file:
            for index, img in enumerate(b2):
                content = ' '.join([str(index), img, shape[0], shape[1]])
                file.write(content)
                file.write('\n')
        with open(sths[2], 'w') as file:
            for index, img in enumerate(b3):
                content = ' '.join([str(index), img, shape[0], shape[1]])
                file.write(content)
                file.write('\n')
        with open(sths[3], 'w') as file:
            for index, img in enumerate(b4):
                content = ' '.join([str(index), img, shape[0], shape[1]])
                file.write(content)
                file.write('\n')
        print('成功切分为四个子集', sths)


if __name__ == '__main__':
    file_type = sys.argv[1]
    file_path = sys.argv[2]
    info_name = sys.argv[3]
    if file_type == 'bin':
        shape1 = sys.argv[4]
        shape2 = sys.argv[5]
        shape = [shape1, shape2]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        get_bin_info(file_path, info_name, shape)
    