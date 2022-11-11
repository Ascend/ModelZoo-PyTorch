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

import torch
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--state-path', type=str)
parser.add_argument('--bin-path', type=str)


def makefile(statepath, outputfile):
    dirs = os.listdir(statepath)
    for file in dirs:
        state = torch.load(statepath+'/'+file)
        mystate = np.array(state).astype(np.float32)
        mystate.tofile(os.path.join(outputfile+'/', file.split('.')[0]+'.bin'))


if __name__ == "__main__":
    args = parser.parse_args()
    state_file = args.state_path
    bin_file = args.bin_path
    makefile(state_file, bin_file)





