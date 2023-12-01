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
    args = parser.parse_args()

    original = Wav2Vec2ForCTC.from_pretrained(args.pytorch_model_dir)   
    #imported = import_huggingface_model(original)  
    input_size = 100000 
    dummy_input = torch.randn(1, input_size, requires_grad=True)  
    model = original.eval()
    traced_model = torch.jit.trace(model.eval(), (dummy_input), strict=False)  
    traced_model.save("traced.pth")
