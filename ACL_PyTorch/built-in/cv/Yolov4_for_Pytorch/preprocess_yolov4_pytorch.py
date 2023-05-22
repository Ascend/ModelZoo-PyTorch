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

import sys
import os
import cv2
import numpy as np
from tqdm import tqdm

def yolov4_onnx(src_info, output_path):
    in_files = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(src_info, 'r') as file:
        contents = file.read().split('\n')
    for i in contents[:-1]:
        in_files.append(i.split()[1])

    for file in tqdm(in_files):
        img0 = cv2.imread(file)
        resized = cv2.resize(img0, (608, 608), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        img_in = img_in.astype(np.int8)
        img_in = np.expand_dims(img_in, axis=0)

        # save img_tensor as binary file for om inference input
        temp_name = file[file.rfind('/') + 1:]
        img_in.tofile(os.path.join(output_path, temp_name.split('.')[0] + ".bin"))


if __name__ == "__main__":
    src_path = os.path.realpath(sys.argv[1])
    save_path = os.path.realpath(sys.argv[2])
    yolov4_onnx(src_path, save_path)
