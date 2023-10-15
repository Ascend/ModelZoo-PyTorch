# Copyright 2021 Huawei Technologies Co., Ltd
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
import torch
import torch_npu
import torch_aie
from tqdm import tqdm


def inference(args):
    img_folder = args.img_folder
    result_folder = args.result_folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    ts_model_path = args.ts_path

    torch_npu.npu.set_device("npu:0")

    torch_script_model = torch.jit.load(ts_model_path).eval()
    compile_input = [torch_aie.Input((1, 3, 1000, 1000), dtype=torch.float)]
    torch_aie_model = torch_aie.compile(torch_script_model, inputs=compile_input)

    for _, __, file_names in os.walk(img_folder):
        for file_name in tqdm(file_names):
            file_path = os.path.join(img_folder, file_name)
            x = torch.from_numpy(np.load(file_path))
            boxes, confidence, key_points_shifts = [
                output.detach().to("cpu").numpy()
                for output in torch_aie_model.forward(x)
            ]
            boxes.tofile(
                os.path.join(result_folder, file_name.replace(".npy", "_2.bin"))
            )
            confidence.tofile(
                os.path.join(result_folder, file_name.replace(".npy", "_1.bin"))
            )
            key_points_shifts.tofile(
                os.path.join(result_folder, file_name.replace(".npy", "_0.bin"))
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--img-folder", default="widerface/prep")
    parser.add_argument("--result-folder", default="result")
    parser.add_argument("--ts-path", default="retinaface.ts")
    args = parser.parse_args()
    inference(args)
