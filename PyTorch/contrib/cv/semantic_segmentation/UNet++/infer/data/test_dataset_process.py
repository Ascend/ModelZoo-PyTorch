# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
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
from glob import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    img_size = 96
    dataset = '../../inputs/dsb2018_%d' % img_size
    
    img_ids = glob(dataset + '/images/*.png')
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    os.makedirs('./input/dsb2018_%d_valid/images' % img_size, exist_ok=True)
    os.makedirs('./input/dsb2018_%d_valid/masks/0' % img_size, exist_ok=True)
    
    for i in range(len(val_img_ids)):
        val_img_id = val_img_ids[i]
        img = cv2.imread(os.path.join(dataset, 'images',
                         val_img_id + '.png'))
        mask = cv2.imread(os.path.join(dataset, 'masks/0',
                         val_img_id + '.png'))
        cv2.imwrite(os.path.join('./input/dsb2018_%d_valid/images' % img_size,
                    val_img_id+'.png'), img)
        cv2.imwrite(os.path.join('./input/dsb2018_%d_valid/masks/0' % img_size,
                    val_img_id+'.png'), mask)


if __name__ == '__main__':
    main()
