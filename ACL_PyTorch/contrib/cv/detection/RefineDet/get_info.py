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
import sys

if __name__ == '__main__':
    im_folder, save_file = sys.argv[1:3]

    im_list = os.listdir(im_folder)
    
    im_list = sorted(im_list)
    with open(save_file, 'w') as f:
        for i, im_name in enumerate(im_list):
            f.write('%d %s 320 320\n'%(i, os.path.join(im_folder, im_name)))
    print('finish!')
