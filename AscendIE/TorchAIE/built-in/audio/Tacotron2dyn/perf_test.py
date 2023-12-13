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
import platform
import argparse
import time
import numpy as np
import torch
import torch_aie



def compute_fps_encoder(model_eval, one_data, num_infer_loops, warm_counter, batch_size, model_name):
    
    default_stream = torch_aie.npu.default_stream()   
    time_cost = 0

    input_1 = one_data[0].to("npu:0")
    input_2 = one_data[1].to("npu:0")
    while warm_counter:
        _ = model_eval(input_1, input_2)
        default_stream.synchronize()
        warm_counter -= 1

    for i in range(num_infer_loops):
        t0 = time.time()
        result = model_eval(input_1, input_2)
        default_stream.synchronize()
        t1 = time.time()
        time_cost += (t1 - t0)
    print(f"{model_name} fps: {num_infer_loops} * {batch_size} / {time_cost : .3f} = {(num_infer_loops * batch_size / time_cost):.3f}samples/s")
    return time_cost

def compute_fps_decoder(model_eval, one_data, num_infer_loops, warm_counter, batch_size, model_name):
    
    default_stream = torch_aie.npu.default_stream()   
    time_cost = 0

    input_1 = one_data[0].to("npu:0")
    input_2 = one_data[1].to("npu:0")
    input_3 = one_data[2].to("npu:0")
    input_4 = one_data[3].to("npu:0")
    input_5 = one_data[4].to("npu:0")
    input_6 = one_data[5].to("npu:0")
    input_7 = one_data[6].to("npu:0")
    input_8 = one_data[7].to("npu:0")
    input_9 = one_data[8].to("npu:0")
    input_10 = one_data[9].to("npu:0")
    input_11 = one_data[10].to("npu:0")

    while warm_counter:
        _ = model_eval(input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11)
        default_stream.synchronize()
        warm_counter -= 1

    for i in range(num_infer_loops):
        t0 = time.time()
        result = model_eval(input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11)
        default_stream.synchronize()
        t1 = time.time()
        time_cost += (t1 - t0)
    print(f"{model_name} fps: {num_infer_loops} * {batch_size} / {time_cost : .3f} = {(num_infer_loops * batch_size / time_cost):.3f}samples/s")
    return time_cost
   
def compute_fps_posnet(model_eval, one_data, num_infer_loops, warm_counter, batch_size, model_name):
    
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
    print(f"{model_name} fps: {num_infer_loops} * {batch_size} / {time_cost : .3f} = {(num_infer_loops * batch_size / time_cost):.3f}samples/s")
    return time_cost


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_model_path', type=str, default='./compiled_models/bs1/encoder_model.ts')
    parser.add_argument('--decoder_model_path', type=str, default='./compiled_models/bs1/decoder_model.ts')
    parser.add_argument('--posnet_model_path', type=str, default='./compiled_models/bs1/posnet_model.ts')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    encoder_model_path = args.encoder_model_path
    decoder_model_path = args.decoder_model_path
    posnet_model_path = args.posnet_model_path
    batch_size = args.batch_size

    loops = 100
    warm_counter = 10
    encoder_model = torch.jit.load(encoder_model_path)

    encoder_input = [torch.zeros((batch_size, 50), dtype = torch.int64), torch.zeros((batch_size,), dtype=torch.int32)]
    time_encoder = compute_fps_encoder(encoder_model, encoder_input, loops, warm_counter, batch_size, "encoder")


    decoder_input = [torch.zeros((batch_size, 80), dtype= torch.float32), 
                torch.zeros((batch_size, 1024), dtype=torch.float32),
                torch.zeros((batch_size, 1024), dtype=torch.float32),
                torch.zeros((batch_size, 1024), dtype=torch.float32),
                torch.zeros((batch_size, 1024), dtype=torch.float32),
                torch.zeros((batch_size, 50), dtype=torch.float32),
                torch.zeros((batch_size, 50), dtype=torch.float32),
                torch.zeros((batch_size, 512), dtype=torch.float32),
                torch.zeros((batch_size, 50, 512), dtype=torch.float32),
                torch.zeros((batch_size, 50, 128), dtype=torch.float32),
                torch.zeros((batch_size, 50), dtype=torch.float32)]
    decoder_model = torch.jit.load(decoder_model_path)
    time_decoder = compute_fps_decoder(decoder_model, decoder_input, loops, warm_counter, batch_size, "decoder")


    posnet_input = torch.zeros((batch_size, 80, 620), dtype= torch.float32)
    posnet_model = torch.jit.load(posnet_model_path)
    time_posnet = compute_fps_posnet(posnet_model, posnet_input, loops, warm_counter, batch_size, "posnet")


    total_time = time_encoder + time_decoder + time_posnet
    print(f" fps: {loops} * {batch_size} / {total_time : .3f} = {(loops * batch_size / total_time):.3f} samples/s")
