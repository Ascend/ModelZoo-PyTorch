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

import argparse
import time
import os

import torch
import torch_aie
import numpy as np
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--aie-module-path", default="./solov2_torchscriptb1_torch_aie.pt")
    parser.add_argument("--batch-size", default=1)
    parser.add_argument("--processed-dataset-path", default="./val2017_bin/")
    parser.add_argument("--output-save-path", default="./result_aie/")
    parser.add_argument("--model-input-height", type=int, default=800, help="input tensor height")
    parser.add_argument("--model-input-width", type=int, default=1216, help="input tensor width")
    parser.add_argument("--device-id", type=int, default=0, help="device id")
    return parser.parse_args()


def load_aie_module(args):
    torch_aie.set_device(args.device_id)
    aie_module = torch.jit.load(args.aie_module_path)
    aie_module.eval()
    return aie_module


def main():
    # Parse user input arguments
    args = parse_arguments()
    if not os.path.exists(args.output_save_path):
        os.makedirs(args.output_save_path)

    # Load AIE module
    aie_module = load_aie_module(args)

    # Start inference
    inference_time = []
    stream = torch_aie.npu.Stream(f"npu:{args.device_id}")
    for idx, filename in enumerate(tqdm(os.listdir(args.processed_dataset_path))):
        file_name = os.path.splitext(filename)[0]
        input_tensor = torch.from_numpy(np.fromfile(args.processed_dataset_path + filename, dtype="float32")).view(1, 3, args.model_input_height, args.model_input_width)
        input_tensor = input_tensor.to(f"npu:{args.device_id}")
        with torch_aie.npu.stream(stream):
            start_time = time.time()
            aie_result = aie_module(input_tensor)
            stream.synchronize()
            cost = time.time() - start_time
            # Warm-up using 5 steps
            if idx >= 5:
                inference_time.append(cost)
        for i, tensor in enumerate(aie_result):
            tensor = tensor.to("cpu")
            tensor.numpy().tofile(f'{args.output_save_path}{file_name}_{i}.bin')

    print(f'\n[INFO] Torch-AIE inference avg cost (batch={args.batch_size}): {sum(inference_time) / len(inference_time) * 1000} ms/pic')


if __name__ == "__main__":
    print("[INFO] SOLOV2 Torch-AIE inference process start")
    main()
