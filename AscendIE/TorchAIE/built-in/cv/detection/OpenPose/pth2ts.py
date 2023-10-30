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
import os
import sys
import torch

sys.path.append("./lightweight-human-pose-estimation.pytorch")
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./weights/checkpoint_iter_370000.pth",
        help="path to the checkpoint",
    )
    parser.add_argument(
        "--ts-path",
        type=str,
        default="./output/human-pose-estimation.ts",
        help="name of output model in torchscript format",
    )
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
    load_state(net, checkpoint)

    net_input = torch.randn(1, 3, 368, 640)
    ts_model = torch.jit.trace(net.eval(), net_input)
    ts_model.save(args.ts_path)


if __name__ == "__main__":
    main()
