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

import argparse
import sys
sys.path.append('./fairseq/')
import fairseq
import torch


def run_pth2onnx(args):
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.model_path])
    model = models[0].to("cpu")
    model.eval()
    source = torch.zeros([1, 580000], dtype=torch.float32).to("cpu")

    input_names = ["source"]
    output_names = ["result"]

    torch.onnx.export(model, source, args.onnx_path, input_names=input_names, output_names=output_names, opset_version=11, verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./data/pt/hubert_large_ll60k_finetune_ls960.pt')
    parser.add_argument('--onnx_path', type=str, default='./hubert.onnx')
    args = parser.parse_args()

    run_pth2onnx(args)


if __name__ == '__main__':
    main()