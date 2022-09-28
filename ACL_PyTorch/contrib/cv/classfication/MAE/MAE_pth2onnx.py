
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import argparse
import sys
from mae import models_vit

def pth2onnx(pth_path,output_file):
    device = torch.device('cpu')
    model = models_vit.__dict__['vit_base_patch16'](
        num_classes=1000,
        drop_path_rate=0.1,
        global_pool=True,
    )
    model.to(device)
    checkpoint = torch.load(pth_path,map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model'],False)
    model.eval()
    input_names = ["image"]
    output_names = ["output"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_file, opset_version=11,
                      input_names=input_names,dynamic_axes=dynamic_axes, output_names=output_names, verbose=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./mae_finetuned_vit_base.pth")
    parser.add_argument('--target', type=str, default="./mae_dynamicbs.onnx")
    args = parser.parse_args()
    pth2onnx(args.source,args.target)