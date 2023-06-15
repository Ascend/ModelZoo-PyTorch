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
sys.path.append("./pycls")
import numpy as np
import cv2
import tqdm

from pycls.datasets import transforms


train_size = 224
test_size = 274

def trans(im):
	im = im[:, :, ::-1].astype(np.float32)
	im = transforms.scale_and_center_crop(im, test_size, train_size)
	im = np.ascontiguousarray(im[:, :, ::-1])
	return im

def EffnetB1_preprocess(src_path, save_path):
	classes = os.listdir(src_path)
	for classname in tqdm.tqdm(classes):
		dirs = os.path.join(src_path, classname)
		save_dir = os.path.join(save_path, classname)
		if not os.path.isdir(save_dir):
			os.makedirs(os.path.realpath(save_dir))
		for img in os.listdir(dirs):
			img_path = os.path.join(dirs, img)
			im = cv2.imread(img_path)
			im = trans(im).astype(np.int8)
			im.tofile(os.path.join(save_dir, img.split('.')[0] + ".bin"))

if __name__ == '__main__':
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    EffnetB1_preprocess(src_path, save_path)