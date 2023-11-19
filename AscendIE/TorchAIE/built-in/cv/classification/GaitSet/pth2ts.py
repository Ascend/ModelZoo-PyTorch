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
import torch
import torch.nn as nn
from GaitSet.model.network import SetNet


class wrapperNet(nn.Module):
    def __init__(self, module):
        super(wrapperNet, self).__init__()
        self.module = module


def main(args):
    encoder = SetNet(args.hidden_dim).float()
    # 增加一层wrapper使得权重keys和模型能够匹配
    encoder = wrapperNet(encoder)
    ckpt = torch.load(args.input_path, map_location=torch.device("cpu"))
    encoder.load_state_dict(ckpt)
    dummy_input = torch.randn(args.input_shape)
    ts_model = torch.jit.trace(encoder.module.eval(), dummy_input)
    ts_model.save(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert pth to torchscript model")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--input_shape", nargs="+", type=int, default=[1, 100, 64, 44])
    parser.add_argument(
        "--input_path",
        default="./GaitSet/work/checkpoint/GaitSet/GaitSet_CASIA-B_73_False_256_0.2_128_full_30-80000-encoder.ptm",
    )
    parser.add_argument("--output_path", default="./gaitset_submit.ts")

    args = parser.parse_args()
    main(args)
