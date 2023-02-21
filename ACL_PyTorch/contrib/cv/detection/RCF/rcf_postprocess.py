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
import numpy as np
import cv2
from scipy.io import savemat

def postprocess_om(args):
    """[om model postprocess]

    Args:
        args ([argparse]): [om model postprocess args]
    """
    if not os.path.exists(args.om_output):
        os.mkdir(args.om_output)
    img_name_list = os.listdir(args.imgs_dir)
    for img_name in img_name_list:
        if img_name.endswith(('jpg', 'png', 'jpeg', 'bmp')):
            img = cv2.imread(os.path.join(args.imgs_dir, '{}'.format(img_name)))
            h, w, c = img.shape
            # Read the output file of the om model
            img_out = np.fromfile('{}/{}_5.bin'.format(args.bin_dir, img_name[:-4]), dtype="float32")
            img_out = img_out.reshape((h, w))
            key = "om_result"
            savemat('{}/{}.mat'.format(args.om_output, img_name[:-4]), {key: img_out})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rcf postprocess') # rcf postprocess parameters
    parser.add_argument('--imgs_dir', default='BSR/BSDS500/data/images/test',
                        type=str, help='images path')
    parser.add_argument('--bin_dir', default='results_bs1',
                        type=str, help='bin file path inferred by benchmark')
    parser.add_argument('--om_output', default='om_out',
                        type=str, help='om postprocess output dir')
    args = parser.parse_args()
    
    postprocess_om(args)
  