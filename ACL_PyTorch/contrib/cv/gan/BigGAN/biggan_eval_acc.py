# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse
import numpy as np
import inception_utils

##############################################################################
# Function
##############################################################################
def get_IS_FID(args):
    # Lists to hold images and labels for images
    img_npz_name = 'gen_img' + '_bs' + str(args.batch_size) + '.npz'
    img = np.load(img_npz_name)['x']
    label_npz_name = 'gen_y' + '.npz'
    label = np.load(label_npz_name)['y']

    # Get Inception Score and FID
    get_inception_metrics = inception_utils.prepare_inception_metrics(args.dataset)

    ################################################################
    # Prepare a simple function get metrics that we use for trunc curves
    def get_metrics():
        IS_mean, IS_std, FID = get_inception_metrics(img, label, args.num_inception_images, num_splits=10, prints=True)
        # Prepare output string
        outstring = 'Noise variance %3.3f, ' % 1.0
        outstring += 'over %d images, ' % args.num_inception_images
        outstring += 'PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (IS_mean, IS_std, FID)
        print(outstring)
    ################################################################

    print('Calculating Inception metrics...')
    get_metrics()


##############################################################################
# Main
##############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="I128")
    parser.add_argument('--num-inception-images', type=int, default=50000)
    opt = parser.parse_args()

    get_IS_FID(opt)
