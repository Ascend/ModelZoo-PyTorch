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

import os
import shutil
import argparse
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))


def ais_infer(ais_infer, model, inputs, outputs):
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")   
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)  
      
    temp_dir_childs_path = os.path.join(temp_dir, "*")
    temp_dir_childs_path_files = os.path.join(temp_dir_childs_path, "*.npy")
            
    for root, dirs, datas in os.walk(inputs):
        for data_name in datas:
            data = os.path.join(inputs, data_name)
            data_npy = np.load(data)
            n, c, h, w = data_npy.shape
            
            rm_cmd = f'rm -rf {temp_dir_childs_path}'
            mv_cmd = f'mv {temp_dir_childs_path_files} {outputs}'
            ais_infer_cmd = f'python3 -m ais_bench --model={model} --input={data} --output={temp_dir} --dymHW={h},{w} --outfmt=NPY'

            if os.system(ais_infer_cmd):
                shutil.rmtree(temp_dir)
                shutil.rmtree(outputs)
                print("bash cmd exect failed: {}".format(ais_infer_cmd))
                return
            
            if os.system(mv_cmd):
                shutil.rmtree(temp_dir)
                shutil.rmtree(outputs)
                print("bash cmd exect failed: {}".format(mv_cmd))
                return
            
            if os.system(rm_cmd):
                shutil.rmtree(temp_dir)
                shutil.rmtree(outputs)
                print("bash cmd exect failed: {}".format(rm_cmd))
                return

    shutil.rmtree(temp_dir)
    print("Infer Results Saved To: {}".format(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ais infer')  # task process paramater
    parser.add_argument('--ais_infer', default='./ais_infer/ais_infer.py', type=str, help='ais_infer.py path')
    parser.add_argument('--model', default='./', type=str, help='om model path')
    parser.add_argument('--inputs', default='./pre_data', type=str, help='input data path')
    parser.add_argument('--batchsize', default='1', type=str, help='om batch size')
    
    args = parser.parse_args()
    
    inputs = os.path.join(args.inputs)
    outputs = os.path.join(__dir__, "results_bs{}".format(args.batchsize))
    if os.path.exists(outputs):
        shutil.rmtree(outputs)
    os.makedirs(outputs)
      
    ais_infer(args.ais_infer, args.model, inputs, outputs)