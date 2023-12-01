# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import sys
import argparse
import torch
import torch_aie
from torch_aie import _enums
import os
import json
import numpy as np
import time


def compute_fps(model_eval, one_data, num_infer_loops, warm_counter, batch_size):
    
    default_stream = torch_aie.npu.default_stream()   
    time_cost = 0
    
    one_data = one_data.to("npu:0")
    while warm_counter:
        _ = model_eval(one_data)
        default_stream.synchronize()
        warm_counter -= 1


    for i in range(num_infer_loops):
        t0 = time.time()
        result = model_eval(one_data)
        default_stream.synchronize()
        t1 = time.time()
        time_cost += (t1 - t0)
    print(f"fps: {num_infer_loops} * {batch_size} / {time_cost:.3f} = {(num_infer_loops * batch_size / time_cost):.3f} samples/s")


def compile(batch_size, torchscript_path):
    torch_aie.set_device(0)
    trace_model = torch.jit.load(torchscript_path)   
    input_info = [torch_aie.Input((batch_size, 100000))]   
    print("compile start")
    pt_model = torch_aie.compile(
        trace_model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        optimization_level=0,
        soc_version="Ascend310P3")
    print("compile success")

    return pt_model


def inference(pt_model, batch_size, input_bin_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    input_filenames = os.listdir(input_bin_path)
    start = time.time()
    batch = 0
    for filename in input_filenames:
        input_file_path = os.path.join(input_bin_path, filename)
        with open(input_file_path, 'rb') as f:
            data = f.read()

        array_data = np.frombuffer(data, dtype=np.float32)  
        tensor_data = torch.tensor(array_data)
        reshaped_tensor = tensor_data.reshape(batch_size, 100000)
        reshaped_tensor = reshaped_tensor.to("npu:0")
        outputs = pt_model.forward(reshaped_tensor)
        outputs = outputs['logits'].to("cpu")
        output_filename = f"output_{filename}.bin"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'wb') as f:
            f.write(outputs.numpy().tobytes())  
        batch = batch + 1
    end = time.time()
    print(f"aie inference end-to-end time: {end - start}")


    #compute QPS
    eg_input_file_path = os.path.join(input_bin_path, "batch_0.bin") 
    with open(eg_input_file_path, 'rb') as f:
        eg_data = f.read()
    eg_array_data = np.frombuffer(eg_data, dtype=np.float32) 
    eg_tensor_data = torch.tensor(eg_array_data)
    reshaped_eg_tensor_data = eg_tensor_data.reshape(batch_size, 100000)  
    loops = 100
    warm_counter = 10
    compute_fps(pt_model, reshaped_eg_tensor_data, loops, warm_counter, batch_size)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',   default=8)
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument ('--torchscript_path')
    args = parser.parse_args()

    batch_size =int(args.batch_size)
    input_bin_path = args.input_path
    output_folder = args.output_path
    torchscript_path = args.torchscript_path

    pt_model = compile(batch_size, torchscript_path)
    inference(pt_model, batch_size, input_bin_path, output_folder)

    
if __name__ == '__main__':
    main()


