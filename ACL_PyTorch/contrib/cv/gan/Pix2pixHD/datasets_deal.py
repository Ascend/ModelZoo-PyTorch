# Copyright 2020 Huawei Technologies Co., Ltd
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
import glob
import shutil

if __name__ == "__main__":
    test_inst_dir = sys.argv[1]   
    test_label_dir = sys.argv[2]   
    test_original_gtfine_dir = sys.argv[3] 

    if not os.path.exists(test_inst_dir):
        os.makedirs(test_inst_dir)
    if not os.path.exists(test_label_dir):
        os.makedirs(test_label_dir)

    city_name_dir = os.listdir(test_original_gtfine_dir)
    img_inst_number = 0
    img_label_number = 0

    for city_name in city_name_dir:
        temp_city_dir = os.path.join(test_original_gtfine_dir, city_name)
        test_gtfine_list = glob.glob(os.path.join(temp_city_dir, "*.png"))

        for img in test_gtfine_list:
            if img[-9:] == "ceIds.png":
                img_inst_number += 1
                shutil.copy(img, test_inst_dir)
            elif img[-9:] == "elIds.png":
                img_label_number += 1
                shutil.copy(img, test_label_dir)
    