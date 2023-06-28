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

import sys

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source='best_model', savedir='best_model')

# Download Thai language sample from Omniglot
class Xvector(torch.nn.Module):
    def __init__(self, model_inp):
        super().__init__()
        self.classifier = model_inp
    
    def forward(self, feats_inp):
        res = self.classifier.feats_classify(feats_inp)
        return res

model = Xvector(classifier)
batch_size=int(sys.argv[1])
feats = torch.randn([batch_size, 1800, 23])

torch.onnx.export(
    model,
    feats,
    'tdnn_bs%d.onnx'%(batch_size),
    input_names=['feats'],
    output_names=['output'],
    export_params=True,
    do_constant_folding=True,
    verbose=True,
    opset_version=11
)