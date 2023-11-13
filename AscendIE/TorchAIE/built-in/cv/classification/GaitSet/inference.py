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


def inference(args):
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder, mode=640)
    torch_npu.npu.set_device(f"npu:{args.device}")

    torch_script_model = torch.jit.load(args.ts_path).eval()
    compile_input = [torch_aie.Input(args.input_shape, dtype=torch.float)]
    torch_aie_model = torch_aie.compile(torch_script_model, inputs=compile_input)

    cumulated_time = 0
    counted_num = 0
    for dir_path, _, file_names in os.walk(args.img_folder):
        for file_name in tqdm(file_names):
            counted_num += 1
            file_path = os.path.join(dir_path, file_name)
            x = torch.from_numpy(
                np.fromfile(file_path, dtype=np.float32).reshape(args.input_shape)
            ).to(f"npu:{args.device}")

            stream = torch_aie.npu.Stream(f"npu:{args.device}")
            with torch_aie.npu.stream(stream):
                stream.synchronize()
                ts = time.time()
                ret = torch_aie_model.forward(x)
                stream.synchronize()
                duration = time.time() - ts
                if counted_num > args.warmup_num:
                    cumulated_time += duration

            feature = ret.detach().to("cpu").numpy()
            feature.tofile(
                os.path.join(args.result_folder, file_name.replace(".bin", "_0.bin"))
            )

    print(
        f"Pure inference performance: {(counted_num - args.warmup_num) / cumulated_time} fps"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup_num", type=int, default=5)
    parser.add_argument("--input_shape", nargs="+", type=int, default=[1, 100, 64, 44])
    parser.add_argument("--ts_path", default="./gaitset_submit.ts")
    parser.add_argument("--img_folder", default="./CASIA-B-bin")
    parser.add_argument("--result_folder", default="./result")
    args = parser.parse_args()
    inference(args)
