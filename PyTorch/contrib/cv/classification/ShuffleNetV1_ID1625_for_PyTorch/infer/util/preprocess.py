# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-3.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import glob
import os
from PIL import Image
import numpy as np
import sys
sys.path.append('..')
from sdk.main import resize
from sdk.main import center_crop
from sdk.main import tranform
from sdk.main import GlobDataLoader
from sdk.main import get_file_name
from sdk.main import parse_args


def main():
    args = parse_args()
    result_fname = get_file_name(args.result_file)
    result_file = f"{result_fname}/"
    if not os.path.exists(result_file):
        os.makedirs(result_file)
    dataset = GlobDataLoader(f"{args.glob}*.JPEG", limit=50000)
    # start preprocess
    for name, _, data in dataset:
        file_path = f"{args.glob}" + name + ".JPEG"
        img = tranform(file_path)
        img.tofile(os.path.join(result_file, name + '.bin'))
        print("-------------------"+name+"-------------------")

    print("success in preprocess")


if __name__ == "__main__":
    main()