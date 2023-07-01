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

import torch
import torch_aie

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmdeploy.apis.torch_jit import trace

if __name__=='__main__':
    deploy_cfg_file = "./mmdeploy/configs/mmocr/text-detection/text-detection_torchscript.py"
    model_cfg_file = "./mmocr/configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py"
    model_file = "./dbnet_resnet50_1200e_icdar2015_20221102_115917-54f50589.pth"
    test_imgs = ["./demo/resources/text_det.jpg"]

    deploy_cfg, model_cfg = load_config(deploy_cfg_file, model_cfg_file)
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    torch_model = task_processor.build_pytorch_model(model_file)
    torch_model.eval()
    print("build model finish")

    raw_inputs, _ = task_processor.create_input(test_imgs)
    inputs = torch_model.data_preprocessor(raw_inputs)['inputs']
    print(f"inputs are: {inputs}")
    print(f"input shape is: {inputs.size()}")

    with torch.no_grad():
        trace_inputs = torch.rand(inputs.size()) / 2
        jit_model = trace(torch_model, trace_inputs, output_path_prefix="dbnet", backend='torchscript')
        print("finish trace")

    aie_input_spec=[
        torch_aie.Input((inputs.size())), # Static NCHW input shape for input #1
    ]

    aie_model = torch_aie.compile(jit_model, inputs=aie_input_spec)

    torch_aie.set_device(0)
    start = time.time()
    print(aie_model(inputs))
    end = time.time()
    print(f"aie time cost {end - start}")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    start = time.time()
    print(jit_model(inputs))
    end = time.time()
    print(f"jit time cost {end - start}")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    with torch.no_grad():
        start = time.time()
        print(torch_model(inputs))
        end = time.time()
        print(f"pytorch time cost {end - start}")