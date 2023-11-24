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

import torch
import numpy as np

from espnet.asr.pytorch_backend.asr import load_trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # load the model
    model, train_args = load_trained_model(args.model_path)
    model.eval()

    inputs = torch.ones((262, 83), dtype=torch.float32)
    mask = None

    output = model.encoder(inputs, mask)
    print(f'output shape: {output.shape}')

    ts_model = torch.jit.trace(model.encoder, inputs)
    ts_model.save('espnet_trace.ts')


if __name__ == "__main__":
    main()
