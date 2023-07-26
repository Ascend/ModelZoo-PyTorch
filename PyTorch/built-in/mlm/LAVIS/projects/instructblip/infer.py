# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
import torch_npu
from PIL import Image

from lavis.models import load_model_and_preprocess
from torch_npu.contrib import transfer_to_npu

torch_npu.npu.set_compile_mode(jit_compile=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer")
    parser.add_argument("--img_path", required=False, default="docs/_static/Confusing-Pictures.jpg", help="image path.")
    parser.add_argument("--prompt", required=False, type=str, default="Describe the image in details.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("npu") if torch_npu.npu.is_available() else "cpu"
    raw_image = Image.open(args.img_path).convert("RGB")
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b",
                                                         is_eval=True, device=device)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    print(model.generate({"image": image, "prompt": args.prompt}, num_beams=2))


if __name__ == "__main__":
    main()