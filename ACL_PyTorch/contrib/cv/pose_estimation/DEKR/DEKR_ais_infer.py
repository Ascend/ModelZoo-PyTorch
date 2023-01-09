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

scale_list = [[512,768], [512, 512], [768, 512], [512, 1024], [1024, 512]]

def ais_infer(bs):
    for i in range(len(scale_list)):
        h, w = scale_list[i][0], scale_list[i][1]
        path = f"out"
        if not os.path.exists(path):
            os.makedirs(path)
        os.system(f'python3 -m ais_bench --model models/dekr_bs{bs}.om --output {path} --dymHW {h},{w} --input prep_data/shape_{h}x{w}')
        for j in os.listdir(path):
            p = os.path.join(path, j)
            if os.path.isdir(p):
                os.system(f'mv {p}/* {path}')
                os.system(f'mv {path}/sumary.json {path}/sumary_{h}x{w}.json')
                os.system(f"rm -rf {p}")

    for i in range(len(scale_list)):
        h, w = scale_list[i][0], scale_list[i][1]
        path = f"out_flip"
        if not os.path.exists(path):
            os.makedirs(path)
        os.system(f'python3 -m ais_bench --model models/dekr_bs{bs}.om --output {path} --dymHW {h},{w} --input prep_data_flip/shape_{h}x{w}')
        for j in os.listdir(path):
            p = os.path.join(path, j)
            if os.path.isdir(p):
                os.system(f'mv {p}/* {path}')
                os.system(f'mv {path}/sumary.json {path}/sumary_{h}x{w}.json')
                os.system(f"rm -rf {p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ais bench')  # task process paramater
    parser.add_argument('--bs', default=1,
                        type=int, help='batchsize')
    args = parser.parse_args()
    ais_infer(args.bs)