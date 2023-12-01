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
import copy

import torch
import torch_aie
import numpy as np
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--aie-module-path", default="./CenterNet_torchscriptb1_torch_aie.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--processed-dataset-path", default="./prep_dataset/")
    parser.add_argument("--output-save-path", default="./result_aie/")
    parser.add_argument("--model-input-height", type=int, default=512, help="input tensor height")
    parser.add_argument("--model-input-width", type=int, default=512, help="input tensor width")
    parser.add_argument("--device-id", type=int, default=0, help="device id")
    parser.add_argument("--warmup-count", type=int, default=5, help="warmup count")
    parser.add_argument("--output-num", type=int, default=3, help="output num")
    return parser.parse_args()


def load_aie_module(args):
    torch_aie.set_device(args.device_id)
    aie_module = torch.jit.load(args.aie_module_path)
    aie_module.eval()
    return aie_module


def get_total_files(args):
    file_paths = []
    for root, _, files in os.walk(args.processed_dataset_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    file_names = [os.path.basename(file_path) for file_path in file_paths]
    total_files = len(file_paths)
    return total_files, file_paths, file_names


def generate_batches(args, total_files, file_paths, file_names):
    batch_size = args.batch_size
    total_batches = (total_files + batch_size - 1) // batch_size
    padding = total_batches * batch_size - total_files

    for batch_num in range(total_batches):
        batch_data = []
        batch_file_names = []
        for item in range(batch_size):
            index = batch_size * batch_num + item
            if index == total_files:
                break
            batch_data.append(
                torch.from_numpy(np.fromfile(file_paths[index], np.float32)).view(
                    [1, 3, args.model_input_height, args.model_input_width]
                )
            )
            batch_file_names.append(file_names[index])
            index += 1
        if (batch_num == (total_batches - 1)) and (padding > 0):
            for _ in range(padding):
                batch_data.append(copy.deepcopy(batch_data[-1]))
                batch_file_names.append(file_names[-1])
        yield torch.cat(batch_data).to(f"npu:{args.device_id}"), batch_file_names


def main():
    print("[INFO] CenterNet Torch-AIE inference process start")

    # Parse user input arguments
    args = parse_arguments()
    if not os.path.exists(args.output_save_path):
        os.makedirs(args.output_save_path)

    # Load AIE module
    aie_module = load_aie_module(args)

    # Generate input data according to batch size
    total_files, file_paths, file_names = get_total_files(args)
    data_generator = generate_batches(args, total_files, file_paths, file_names)

    # Start inference
    inference_time = []
    stream = torch_aie.npu.Stream(f"npu:{args.device_id}")

    for count, (input_tensor, batched_file_name) in enumerate(tqdm(data_generator, total=total_files), start=1):
        input_tensor = input_tensor.to(f"npu:{args.device_id}")
        with torch_aie.npu.stream(stream):
            start_time = time.time()
            aie_result = aie_module(input_tensor)
            stream.synchronize()
            cost = time.time() - start_time
            # Warm-up using 5 steps by default
            if count >= args.warmup_count:
                inference_time.append(cost)

        for i, file_name in enumerate(batched_file_name):
            file_name = file_name.split('.')[0]

            for j in range(args.output_num):
                aie_result_j = aie_result[j].to("cpu")
                aie_result_j[i].numpy().tofile(f'{args.output_save_path}{file_name}_{j}.bin')

    # Calculate inference avg cost and throughput
    aie_avg_cost = sum(inference_time) / len(inference_time) * 1000
    aie_throughput = args.batch_size / (sum(inference_time) / len(inference_time))

    print(f'\n[INFO] Torch-AIE inference avg cost (batch={args.batch_size}): {aie_avg_cost} ms/pic')
    print(f'[INFO] Throughput = {aie_throughput} pic/s')
    print('[INFO] CenterNet Torch-AIE inference process finished')


if __name__ == "__main__":
    main()
