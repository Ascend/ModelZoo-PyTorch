# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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

import cv2
import torch
import torch_npu
import numpy as np

import llama
from torch_npu.contrib import transfer_to_npu
from ais_bench.infer.interface import InferSession

DEVICE_ID = 0
DEVICE = "cpu"
IMG_SIZE = 224
BATCH_SIZE = 5
INFER_LOOP = 5
LLAMA_DIR = "/path/to/LLaMA/"
CLIP_DIR = "/path/to/clip/"
BIAS_DIR = "/path/to/BIAS-7B"
PIC_FILE_PATH = "/path/to/picture"

torch.npu.set_device(torch.device(f"npu:{DEVICE_ID}"))

option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

# load clip om model
clip_model = InferSession(DEVICE_ID, CLIP_DIR)

# choose from BIAS-7B, LORA-BIAS-7B
model = llama.load(BIAS_DIR, LLAMA_DIR, DEVICE)
model.eval().npu()

# choose device type
soc_version = torch_npu._C._npu_get_soc_version()
if soc_version in [104, 220, 221, 222, 223]:
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.npu_format_cast(2)
else: 
    # if on 910A or 310P chip, eliminate the TransData and 
    # Transpose ops by converting weight data types 
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == 'lm_head':
                # eliminate TransData op before lm_head calculation
                module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
            module.weight.data = module.weight.data.npu_format_cast(29)

for _, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = module.weight.data.npu_format_cast(2)


def img_process(file_path, begin_idx=1, bsz=5):
    res = None
    for i_ in range(begin_idx, begin_idx + bsz):
        img_name = f"pic_{i_}.jpg"
        img = cv2.imread(file_path + img_name)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        res = img if res is None else np.concatenate((res, img), axis=0)
    res = res.reshape(bsz, *img.shape)
    res = res.astype(np.float16)

    return res
    
model.init_acl_weight()

prompt = llama.format_prompt("Is there a fight in the picture?")
prompt_bsz = []
for i in range(BATCH_SIZE):
    prompt_bsz.append(prompt)

inputs_warm = img_process(PIC_FILE_PATH, 1, BATCH_SIZE)

with torch.no_grad():
    result_warm = model.generate(clip_model, inputs_warm, prompt_bsz)
    print("[result]:", result_warm)

sum_time = 0
for i in range(INFER_LOOP):
    start_time = time.time()
    inputs_bsz = img_process(PIC_FILE_PATH, i * BATCH_SIZE + 1, BATCH_SIZE)
    with torch.no_grad():
        result = model.generate(clip_model, inputs_bsz, prompt_bsz)
    end_time = time.time()
    cur_time = (end_time - start_time) * 1000
    sum_time += cur_time
    print("[result]:", result)
    print("[tiral time]", cur_time)

avg_time = sum_time / INFER_LOOP
print("[avg_time]:", avg_time)

