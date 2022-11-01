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
import torch.onnx # https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
from transformers import Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_model_dir',
                            help='directory for Pytorch model and configuration files',
                            default="./wav2vec2_pytorch_model/")
    parser.add_argument('--output_model_path',
                            help='the output path of ONNX model',
                            default="./wav2vec2.onnx")
    args = parser.parse_args()

    original = Wav2Vec2ForCTC.from_pretrained(args.pytorch_model_dir)
    imported = import_huggingface_model(original)

    input_size = 100000  # audio max len

    dummy_input = torch.randn(1, input_size, requires_grad=True)
    torch.onnx.export(imported,                 # model being run
                    dummy_input,              # model input (or a tuple for multiple inputs)
                    args.output_model_path,        # where to save the model
                    export_params=True,       # store the trained parameter weights inside the model file
                    opset_version=11,                 # the ONNX version to export the model to
                    do_constant_folding=True,         # whether to execute constant folding for optimization
                    input_names=['modelInput'],       # the model's input names
                    output_names=['modelOutput'],     # the model's output names
                    dynamic_axes={'modelInput': {0: 'batch_size'},    # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
