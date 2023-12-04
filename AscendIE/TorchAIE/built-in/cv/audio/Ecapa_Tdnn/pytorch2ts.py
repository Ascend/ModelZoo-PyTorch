# Copyright 2022 Huawei Technologies Co., Ltd
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

import torch

import sys

from ECAPA_TDNN.main import ECAPA_TDNN, load_checkpoint
from torch import optim
from functools import partial


def pth2onnx(checkpoint, output_file):
    device = torch.device('cpu')
    model = ECAPA_TDNN(1211, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-5)
    model, optimizer, step = load_checkpoint(model, optimizer, checkpoint, rank='cpu')
    model.forward = partial(model.forward, infer=True)
    # 调整模型为eval mode
    model.eval()
    dummy_input1 = torch.randn(1, 80, 200).to(device)

    ts_model = torch.jit.trace(model, dummy_input1)
    ts_model.save(output_file)


if __name__ == "__main__":
    checkpoint = sys.argv[1]
    save_path = sys.argv[2]
    pth2onnx(checkpoint, save_path)
