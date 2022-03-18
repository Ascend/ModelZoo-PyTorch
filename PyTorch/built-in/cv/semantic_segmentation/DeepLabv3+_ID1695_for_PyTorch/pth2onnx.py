# coding: utf-8
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

from collections import OrderedDict

import torch.onnx

from modeling.deeplab import DeepLab


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(pth_path, onnx_path, num_classes, backbone="resnet",
            output_stride=16):
    model = DeepLab(num_classes=num_classes,
                    backbone=backbone,
                    output_stride=output_stride,
                    sync_bn=False,
                    freeze_bn=False)
    print("init model success")
    checkpoint = torch.load(pth_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')

    print(f"{pth_path} best_pred is {checkpoint['best_pred']}")
    model.load_state_dict(checkpoint['state_dict'])
    print("load state dict success")
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 513, 513)
    dynamic_axes = {'actual_input_1':{0:'-1'},'output1':{0:'-1'}}
    print("begin to export")
    torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names,
                      output_names=output_names, dynamic_axes = dynamic_axes,
                      opset_version=11)
    print(f"export {pth_path} to {onnx_path} success.")


if __name__ == "__main__":
    num_classes = 21
    backbone = "resnet"
    pth_path = "model_best.pth.tar"
    onnx_path = "deeplabv3plus.onnx"
    convert(pth_path, onnx_path, num_classes, backbone)
