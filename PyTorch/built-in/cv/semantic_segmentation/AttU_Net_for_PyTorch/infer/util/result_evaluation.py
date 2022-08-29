# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
import sys

sys.path.append('../..')
import numpy as np
from scipy.special import expit
from infer_evaluation import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--res_path', type=str, default="./result.txt")
    config = parser.parse_args()

    res_image = np.loadtxt(config.res_path)
    test_GT_images = np.loadtxt("../data/input/valid_GT_images.txt")
    image_size = res_image.shape[0]
    threshold = 0.5

    acc = 0.  # Accuracy
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    F1 = 0.  # F1 Score
    JS = 0.  # Jaccard Similarity
    DC = 0.  # Dice Coefficient
    length = 0

    for i in range(image_size):
        res = res_image[i].reshape(1, 1, 224, 224).astype(np.float32)
        test_GT_image = test_GT_images[i].reshape(1, 1, 224, 224).astype(np.float32)

        SR = expit(res)
        SR_ac = SR > threshold
        GT_ac = test_GT_image == np.max(test_GT_image)
        acc += get_accuracy(SR_ac, GT_ac)
        SE += get_sensitivity(SR_ac, GT_ac)
        SP += get_specificity(SR_ac, GT_ac)
        PC += get_precision(SR_ac, GT_ac)
        F1 += get_F1(SR_ac, GT_ac)
        JS += get_JS(SR_ac, GT_ac)
        DC += get_DC(SR_ac, GT_ac)
        length += 1

    acc = acc / length
    SE = SE / length
    SP = SP / length
    PC = PC / length
    F1 = F1 / length
    JS = JS / length
    DC = DC / length
    unet_score = JS + DC
    print("Test finished, model acc: %.3f " % (acc))
