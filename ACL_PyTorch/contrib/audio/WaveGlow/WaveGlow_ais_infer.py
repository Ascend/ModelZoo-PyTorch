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
import argparse

scale_list = [['LJ001-0001.wav.bin', 832], ['LJ001-0002.wav.bin', 164], ['LJ001-0003.wav.bin', 833], ['LJ001-0004.wav.bin', 443], \
    ['LJ001-0005.wav.bin', 699], ['LJ001-0006.wav.bin', 490], ['LJ001-0007.wav.bin', 723], ['LJ001-0008.wav.bin', 154],\
     ['LJ001-0009.wav.bin', 651], ['LJ001-0010.wav.bin', 760]]

def ais_infer(bs, ais_infer_path, om_model):
    for i in range(len(scale_list)):
        file_path, scale = scale_list[i][0], scale_list[i][1]
        path = f"out"
        if not os.path.exists(path):
            os.makedirs(path)
        os.system(f'python3 -m ais_bench --input prep_data/{file_path} --dymDims mel:1,80,{scale} --model {om_model} --output {path} --outfmt BIN --batchsize {bs}')
        for j in os.listdir(path):
            p = os.path.join(path, j)
            if os.path.isdir(p):
                os.system(f'mv {p}/* {path}')
                os.system(f'mv {path}/sumary.json {path}/sumary_{i}.json')
                os.system(f"rm -rf {p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ais infer')  # task process paramater
    parser.add_argument('--ais_infer_path', default='ais_infer', type=str)
    parser.add_argument('--bs', default=1,
                        type=int, help='batchsize')
    parser.add_argument('--om_model', default='WaveGlow.om',
                        type=str, help='om file')
    args = parser.parse_args()
    ais_infer(args.bs, args.ais_infer_path, args.om_model)