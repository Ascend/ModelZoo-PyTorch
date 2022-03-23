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

# 3d_nested_unet_postprocess.py
import sys
import os
import time
import pdb
import argparse
from nnunet.inference import predict_simple2


def main():
    # pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--file_path', help='output bin files path', required=True)
    args = parser.parse_args()
    python_file = predict_simple2.__file__  # /home/hyp/UNetPlusPlus/pytorch/nnunet/inference/predict_simple2.py
    file_path = args.file_path
    pre_mode = 2
    command = 'python3 ' + str(python_file) + ' --pre_mode ' + str(pre_mode) + ' --file_path ' + str(file_path)
    os.system(command)


if __name__ == "__main__":
    main()
    print('main end')

