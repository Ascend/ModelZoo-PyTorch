"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
import torch
from jasper import config
from common import helpers
from jasper.model import Jasper


def pth2onnx(ckpt_path, out_path):
    cfg = config.load('configs/jasper10x5dr_speedp-online_speca.yaml')
    model = Jasper(encoder_kw=config.encoder(cfg),
                   decoder_kw=config.decoder(cfg, n_classes=29))
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = helpers.convert_v1_state_dict(checkpoint['ema_state_dict'])
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    feats = torch.randn([1, 64, 4000], dtype=torch.float32)
    feat_lens = torch.tensor([1], dtype=torch.int32)
    dynamic_axes = {'feats': {0: '-1'}, 'feat_lens': {0: '-1'}, 
                    'prob': {0: '-1'}, 'label': {0: '-1'}}
    torch.onnx.export(model,
                      (feats, feat_lens),
                      out_path,
                      input_names=['feats', 'feat_lens'],
                      output_names=['prob', 'label'],
                      dynamic_axes=dynamic_axes,
                      opset_version=11)


if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    out_path = sys.argv[2]
    pth2onnx(ckpt_path, out_path)
