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
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", default="environment/", help='the path of folder RESULTS_FOLDER')
    parser.add_argument("-mode", default="1p", help='1p or 8p')
    parser.add_argument("-target", default="Dice", help='The target')
    args = parser.parse_args()
    print('args=', args)
    print('----------')
    path = str(args.path)
    mode = str(args.mode)
    target = str(args.target)  # this is 'Dice'
    input_file = None
    if mode == '1p':
        input_file = 'RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json'
    elif mode == '8p':
        input_file = 'RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2_hypDDP__nnUNetPlansv2.1/fold_0/validation_raw/summary.json'
    else:
        print('Get wrong mode. You should set mode to 1p or 8p.')
        return
    input_file = os.path.join(path, input_file)

    print('Ready to read .json file:', input_file)
    dice_1, dice_2 = 0.0, 0.0
    with open(input_file, 'r') as f:
        dic = json.load(f)
        results = dic['results']
        mean = results['mean']
        dice_1 = mean['1'][target]
        dice_2 = mean['2'][target]
    dice_1, dice_2 = float(dice_1)*100, float(dice_2)*100
    print('The Dice result:')
    print('Liver 1_Dice (val):', dice_1)
    print('Liver 2_Dice (val):', dice_2)


if __name__ == "__main__":
    main()
