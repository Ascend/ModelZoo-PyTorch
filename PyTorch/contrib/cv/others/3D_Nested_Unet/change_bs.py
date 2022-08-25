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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", default="environment/", help='the path of folder nnUNet_preprocessed')
    parser.add_argument("-size", default="1", help='the batchsize value')
    args = parser.parse_args()
    print('args=', args)
    print('----------')
    path = str(args.path)
    size = int(args.size)
    assert size >= 1
    input_file = 'nnUNet_preprocessed/Task003_Liver/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = 'nnUNet_preprocessed/Task003_Liver/nnUNetPlansv2.1_plans_3D.pkl'
    input_file = os.path.join(path, input_file)
    output_file = os.path.join(path, output_file)

    print('Ready to read .pkl file:', input_file)
    a = load_pickle(input_file)
    batchsize_old = int(a['plans_per_stage'][0]['batch_size'])
    print('The batchsize is', batchsize_old, 'before modification')
    print('Ready to change batchsize to', size)
    a['plans_per_stage'][0]['batch_size'] = size
    a['plans_per_stage'][1]['batch_size'] = size
    print('Batchsize: ', batchsize_old, '->->->', size)
    print('Modification completed! Rewrite the file:', output_file)
    save_pickle(a, output_file)


if __name__ == "__main__":
    main()
