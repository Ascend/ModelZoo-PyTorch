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
import torch.onnx
sys.path.append('./wseg')
from wseg.core.config import cfg, cfg_from_file
from wseg.models import get_model

def add_model(name, model):
    models = {}
    models[name] = {}
    models[name]['model'] = model
    return models


def load(paths, models):
    for _, data in models.items():
        if data['model'] is not None:
            data['model'].load_state_dict(torch.load(paths,map_location=torch.device('cpu')))
    print("load weight from", paths)


def pth2onnx(cfg_file, pth_file, output_file):
    cfg_from_file(cfg_file)

    model = get_model(cfg.NET, num_classes=cfg.TEST.NUM_CLASSES)
    models = add_model('enc', model)
    load(pth_file, models)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    input_names = ["image"]
    output_names = ["cls", "mask"]
    dynamic_axes = {'image': {0: '-1'}, 'cls': {0: '-1'}, 'mask': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 1024, 1024)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, verbose=True, opset_version=11)

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    pth_path = sys.argv[2]
    output_path = sys.argv[3]
    pth2onnx(cfg_path, pth_path, output_path)

