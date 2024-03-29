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

import time
from tqdm import tqdm

import torch
import torch_aie


def forward_infer(model, dataloader, batchsize, device_id):
    pred_results = []
    inference_time = []
    loop_num = 0
    for snd in tqdm(dataloader):
        result, inference_time = pt_infer(model, snd[0].to(torch.float32), snd[1], device_id, loop_num, inference_time)
        pred_results.append(result)
        loop_num += 1

    avg_inf_time = sum(inference_time) / len(inference_time) / batchsize * 1000
    print('performance(ms)：', avg_inf_time)
    print("throughput(fps): ", 1000 / avg_inf_time)

    return pred_results

def pt_infer(model, input_li_1, input_li_2, device_id, loop_num, inference_time):

    input_npu_li_1 = input_li_1.to("npu:" + str(device_id))
    input_npu_li_2 = input_li_2.to("npu:" + str(device_id))
    stream = torch_aie.npu.Stream("npu:" + str(device_id))
    with torch_aie.npu.stream(stream):
        inf_start = time.time()
        output_npu = model.forward(input_npu_li_1, input_npu_li_2)
        stream.synchronize()
        inf_end = time.time()
        inf = inf_end - inf_start
        if loop_num >= 5:   # use 5 step to warmup
            inference_time.append(inf)

    results = [output_npu[0].to("cpu"), output_npu[1].to("cpu")]
    return results, inference_time
