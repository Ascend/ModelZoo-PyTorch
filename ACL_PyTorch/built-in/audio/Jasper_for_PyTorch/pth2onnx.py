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

import torch
from jasper import config
from common import helpers
from jasper.model import Jasper


def main():
    cfg = config.load('configs/jasper10x5dr_speedp-online_speca.yaml')
    model = Jasper(encoder_kw=config.encoder(cfg),
                   decoder_kw=config.decoder(cfg, n_classes=29))
    checkpoint = torch.load('checkpoints/jasper_fp16.pt', map_location="cpu")
    state_dict = helpers.convert_v1_state_dict(checkpoint['ema_state_dict'])
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    feats = torch.randn([4, 64, 4000], dtype=torch.float32)
    feat_lens = torch.tensor([1000], dtype=torch.int32)
    dynamic_axes = {'feats': {2: '-1'}, 'output': {1: '-1'}}
    torch.onnx.export(model,
                      (feats, feat_lens),
                      'jasper_dynamic.onnx',
                      input_names=['feats', 'feat_lens'],
                      output_names=['output'],
                      dynamic_axes=dynamic_axes,
                      verbose=True,
                      opset_version=11)


if __name__ == '__main__':
    main()
