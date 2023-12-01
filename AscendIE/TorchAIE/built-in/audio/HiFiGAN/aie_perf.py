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

import argparse
import torch
import torch_aie
import time

def inference(batch_size, mel_len, model):
    infer_time  = []
    inf_stream = torch_aie.npu.Stream("npu:0")
    for i in range(200):
        inputs = torch.zeros((batch_size, 80, 1, mel_len))
        inputs_npu = inputs.to("npu:0")

        inf_s = time.time()
        with torch_aie.npu.stream(inf_stream):
            result = model(inputs_npu)
        inf_stream.synchronize()
        inf_e = time.time()
        infer_time.append(inf_e - inf_s)

    avg_inf_time = sum(infer_time[3:]) / len(infer_time[3:])
    throughput = batch_size / avg_inf_time
    print("torch_aie qps is : ", throughput)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mel_len', type=int, default=250)
    parser.add_argument('--aie_dir', type=str, default='./aie_model.ts')
    args = parser.parse_args()
    torch_aie.set_device(0)
    model = torch.jit.load(args.aie_dir)
    inference(args.batch_size, args.mel_len, model)
