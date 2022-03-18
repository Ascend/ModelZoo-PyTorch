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
from PIL import Image
import numpy as np
import multiprocessing


model_config = {
    'vgg16_ssd': {
        'resize': 300,
        'mean': [123, 117, 104],
        'std': [1., 1., 1.],
    },
}


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        return img.resize((size, size), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def gen_input_bin(mode_type, file_batches, batch):
    i = 0
    for file in file_batches[batch]:
        i = i + 1
        print("batch", batch, file, "===", i)

        # RGBA to RGB
        image = Image.open(os.path.join(src_path, file)).convert('RGB')
        image = resize(image, model_config[mode_type]['resize']) # Resize
        img = np.array(image, dtype=np.float32)
        img -= np.array(model_config[mode_type]['mean'], dtype=np.float32) # Normalize: mean
        img /= np.array(model_config[mode_type]['std'], dtype=np.float32) # Normalize: std
        img = img.transpose(2, 0, 1) # ToTensor: HWC -> CHW
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


def preprocess(mode_type, src_path, save_path):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 500] for i in range(0, 50000, 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(mode_type, file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure preprocess succcess.")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception("usage: python3 xxx.py [model_type] [src_path] [save_path]")
    mode_type = sys.argv[1]
    src_path = sys.argv[2]
    save_path = sys.argv[3]
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)
    if mode_type not in model_config:
        model_type_help = "model type: "
        for key in model_config.keys():
            model_type_help += key
            model_type_help += ' '
        raise Exception(model_type_help)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    preprocess(mode_type, src_path, save_path)
