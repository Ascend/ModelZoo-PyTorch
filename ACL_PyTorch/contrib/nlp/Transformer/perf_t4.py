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
import time
import torch
import argparse
import numpy as np
import onnxruntime as rt
print(rt.get_device())

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--path", default="./transformer710_final.onnx", type=str, required=True)
    args = parser.parse_args()
    ort_session = rt.InferenceSession(args.path, providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    inf_dummy_input = torch.ones(1, 15).long()
    inf_dummy_input = inf_dummy_input.numpy()
    totaltime = 0
    count = 50
    for i in range(count):
        print('i ', i)
        t1 = time.time() # time_sync()
        pred_onnx = ort_session.run(None, {input_name: inf_dummy_input})
        t2 = time.time() # time_sync()
        totaltime += (t2 - t1)
    onnx_meantime = totaltime / count 
    fps = args.bs / onnx_meantime
    print('transformer inference by onnxruntime mean time is {}s, bs{}, {}fps'.format(onnx_meantime, args.bs, fps))