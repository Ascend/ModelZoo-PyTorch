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

import torch

import argparse
import torch_aie
from torch_aie import _enums

import os

import json
import numpy as np
import time


def compile(batch_size, encoder_path, decoder_path, posnet_path, waveglow_path, output_path):
    torch_aie.set_device(0)

    encoder_input_info = [torch_aie.Input((batch_size, 50), dtype=torch.int64), torch_aie.Input((batch_size,), dtype=torch.int32)]
    encoder_trace_model = torch.jit.load(encoder_path)

    print("load traced_encoder")
    print("compile start")
    encoder_pt_model = torch_aie.compile(
        encoder_trace_model,
        inputs=encoder_input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        optimization_level=0,
        soc_version="Ascend310P3")
    encoder_output_path = os.path.join(output_path, "encoder_model.ts")
    encoder_pt_model.save(encoder_output_path)
    print("compile success")

    decoder_input_info = [torch_aie.Input((batch_size, 80), dtype= torch.float32), 
                torch_aie.Input((batch_size, 1024), dtype=torch.float32),
                torch_aie.Input((batch_size, 1024), dtype=torch.float32),
                torch_aie.Input((batch_size, 1024), dtype=torch.float32),
                torch_aie.Input((batch_size, 1024), dtype=torch.float32),
                torch_aie.Input((batch_size, 50), dtype=torch.float32),
                torch_aie.Input((batch_size, 50), dtype=torch.float32),
                torch_aie.Input((batch_size, 512), dtype=torch.float32),
                torch_aie.Input((batch_size, 50, 512), dtype=torch.float32),
                torch_aie.Input((batch_size, 50, 128), dtype=torch.float32),
                torch_aie.Input((batch_size, 50), dtype=torch.float32)]

    decoder_trace_model = torch.jit.load(decoder_path)
    print("load traced_decoder")
    decoder_pt_model = torch_aie.compile(
        decoder_trace_model,
        inputs=decoder_input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        optimization_level=0,
        soc_version="Ascend310P3")
    decoder_output_path = os.path.join(output_path, "decoder_model.ts")
    decoder_pt_model.save(decoder_output_path)
    print("compile success")
   
    posnet_input_info = [torch_aie.Input((batch_size, 80, 620), dtype= torch.float32)]#original
    posnet_trace_model = torch.jit.load(posnet_path)  
    print("load traced_postnet")
    posnet_pt_model = torch_aie.compile(
        posnet_trace_model,
        inputs=posnet_input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        optimization_level=0,
        soc_version="Ascend310P3")
    posnet_output_path = os.path.join(output_path, "posnet_model.ts")
    posnet_pt_model.save(posnet_output_path)
    print("compile success")
    
    waveglow_input_info = [torch_aie.Input((batch_size, 80, 620), dtype = torch.float32), torch_aie.Input((batch_size, 8, 19840), dtype = torch.float32)]
    waveglow_trace_model = torch.jit.load(waveglow_path)  
    print("load traced_waveglow")
    
    print("compile start")
    waveglow_pt_model = torch_aie.compile(
        waveglow_trace_model,
        inputs=waveglow_input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        optimization_level=0,
        soc_version="Ascend310P3")
    print("compile over")
    waveglow_output_path = os.path.join(output_path, "waveglow_model.ts")
    waveglow_pt_model.save(waveglow_output_path)
    print("compile success")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, required=True)
    parser.add_argument('--decoder_path', type=str, required=True)
    parser.add_argument('--posnet_path', type=str, required=True)
    parser.add_argument('--waveglow_path', type=str, required=True)
    parser.add_argument('--batchsize', type=int, required=True)
    parser.add_argument('--compiled_models_folder', type=str, required=True)
    args = parser.parse_args()

    batch_size = int(args.batchsize)
    output_path = os.path.join(args.compiled_models_folder, "bs{}".format(batch_size))  
    os.makedirs(output_path, exist_ok=True)
    pt_model = compile(batch_size, args.encoder_path, args.decoder_path, args.posnet_path, args.waveglow_path, output_path)

    torch_aie.finalize()


if __name__ == '__main__':
    main()