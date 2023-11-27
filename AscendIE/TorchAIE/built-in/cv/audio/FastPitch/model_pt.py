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

import torch
import torch_aie
import time
import numpy as np
from tqdm import tqdm

def forward_nms_script(model, dataloader, batchsize, device_id):
    pred_results = []
    inference_time = []
    loop_num = 0
    for img in tqdm(dataloader):
        # print(torch.tensor(img).shape)
        img_input = torch.tensor([i[0].float().numpy().tolist() for i in img])
        # print("input:", img_input.shape)
        # pt infer
        result, inference_time = pt_infer(model, img_input, device_id, loop_num, inference_time)
        pred_results.append(result)
        loop_num += 1

    # print(batchsize, inference_time)
    avg_inf_time = sum(inference_time) / len(inference_time) / batchsize * 1000
    print('性能(毫秒)：', avg_inf_time)
    print("throughput(fps): ", 1000 / avg_inf_time)
    # print("0", pred_results[0][0].shape)
    # print("1", pred_results[0][1].shape)
    # print("2", pred_results[0][2].shape)
    # print("3", pred_results[0][3].shape)
    # print("4", pred_results[0][4].shape)
    return pred_results

def pt_infer(model, input_li, device_id, loop_num, inference_time):
    input_npu_li = input_li.to("npu:" + str(device_id))
    stream = torch_aie.npu.Stream("npu:" + str(device_id))
    with torch_aie.npu.stream(stream):
        inf_start = time.time()
        output_npu = model.forward(input_npu_li)
        stream.synchronize()
        inf_end = time.time()
        inf = inf_end - inf_start
        if loop_num >= 5:   # use 5 step to warmup
            inference_time.append(inf)
    results = tuple([i.to("cpu") for i in output_npu])
    # t = torch.tensor(results[0])
    # print(len(results))
    # print(t.shape)
    # print("results:", results)
    # print("shape:", np.array(results).shape)
    return results, inference_time
