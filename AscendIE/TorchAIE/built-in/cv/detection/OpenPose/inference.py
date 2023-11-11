# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np
import os
import time
import torch
import torch_npu
import torch_aie
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--processed-img-path", default="datasets/coco/processed_img")
    parser.add_argument("--result-folder", default="output/aie_inference_result")
    parser.add_argument("--ts-path", default="output/human-pose-estimation.ts")
    args = parser.parse_args()

    processed_img_path = args.processed_img_path
    result_folder = args.result_folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    ts_model_path = args.ts_path

    torch_npu.npu.set_device("npu:0")

    torch_script_model = torch.jit.load(ts_model_path)
    compile_input = [torch_aie.Input((1, 3, 368, 640), dtype=torch.float)]
    print("Start compiling the aie model...")
    torch_aie_model = torch_aie.compile(torch_script_model, inputs=compile_input)
    print("Compilation finished.")

    cumulated_time = 0
    warmup_num = 5
    counted_num = 0
    for dir_path, _, file_names in os.walk(processed_img_path):
        for file_name in tqdm(file_names):
            counted_num += 1
            file_path = os.path.join(dir_path, file_name)
            x = (
                torch.from_numpy(np.fromfile(file_path, np.float32))
                .view(1, 3, 368, 640)
                .to("npu:0")
            )

            stream = torch_aie.npu.Stream("npu:0")
            with torch_aie.npu.stream(stream):
                stream.synchronize()
                ts = time.time()
                ret = torch_aie_model.forward(x)
                stream.synchronize()
                duration = time.time() - ts
                if counted_num > warmup_num:
                    cumulated_time += duration

            _, __, pcm, paf = [output.detach().to("cpu").numpy() for output in ret]
            pcm.tofile(os.path.join(result_folder, file_name.replace(".bin", "_0.bin")))
            paf.tofile(os.path.join(result_folder, file_name.replace(".bin", "_1.bin")))

    print(
        f"Pure inference performance: {(counted_num - warmup_num) / cumulated_time} fps"
    )


if __name__ == "__main__":
    main()
