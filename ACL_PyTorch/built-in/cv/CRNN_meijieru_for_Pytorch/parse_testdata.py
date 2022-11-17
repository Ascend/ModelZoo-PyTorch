# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import sys
import dataset
from PIL import Image
import numpy as np


def gen_data_label(img_path, data_dir):
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(img_path).convert("L")
    image = transformer(image)
    image = image.view(1, *image.size())
    image = np.array(image, np.float32)
    image.tofile(f'{data_dir}/demo.bin')


if __name__ == '__main__':
    img_path = sys.argv[1]
    data_dir = sys.argv[2]

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    gen_data_label(test_dir, output_bin)
