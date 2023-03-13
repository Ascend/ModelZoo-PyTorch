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
import sys
import torch

sys.path.append('../../src/utils/')
sys.path.append('../../src/transformer/')
from transformer import Transformer


parser = argparse.ArgumentParser("Speech-Transformer-pth2onnx-decoder")
parser.add_argument('--pth-path', type=str,  default='./final.pth.tar')
parser.add_argument('--tgt-word-prj-path', type=str,
                    default='./tgt_word_prj.onnx')


def pth2onnx(model, output_file, input):
    model = model.to('cpu')
    model.eval()
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, input, output_file, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True)


def main():
    args = parser.parse_args()
    model_path = args.pth_path
    output_path = args.tgt_word_prj_path
    model, _, _ = Transformer.load_model(model_path)
    tgt_word_prj = (model.decoder.tgt_word_prj)
    dummy_input = torch.rand(size=(1, 512), dtype=torch.float32)
    pth2onnx(tgt_word_prj, output_path, dummy_input)


if __name__ == "__main__":
    main()
