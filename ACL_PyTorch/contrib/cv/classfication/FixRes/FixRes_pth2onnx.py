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
import torch
import torch.onnx
import onnxruntime
import numpy as np
import onnx
import torchvision.models as models

parser = argparse.ArgumentParser(description="FixRes pth2onnx")
parser.add_argument('--model', default='FixRes', type=str, metavar='MODEL',
                        help='Name of model to train (default: "countception"')
parser.add_argument("--pretrain_path", default="./ResNetFinetune.pth", type=str)
parser.add_argument("--output_name", default="./FixRes.onnx", type=str)
args = parser.parse_args()

def pth2onnx(model, onnx_name):
    """Export onnx from pytorch model

    Args:
        model (Model): pytorch model to export
    """
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {"image": {0: "-1"}, "class": {0: "-1"}}
    dummy_input_bs1 = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input_bs1, onnx_name, input_names=input_names,\
        dynamic_axes=dynamic_axes, output_names=output_names, opset_version=11, verbose=True)


def to_numpy(tensor):
    """convert tensor to ndarray

    Args:
        tensor (torch.Tensor): tensor to be converted

    Returns:
        [ndarray]: ndarray converted from tensor
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    """Main function to export onnx from pytorch model

    Raises:
        ValueError: pretrained path not assigned
    """
    model = models.resnet50(pretrained=False)
    if args.pretrain_path is None: 
        raise ValueError("pretrain path required for onnx")
    pretrained_dict = torch.load(args.pretrain_path, map_location='cpu')['model']
    model_dict = model.state_dict()
    count = 0
    count2 = 0
    for k in model_dict.keys():
        count = count + 1.0
        if(('module.' + k) in pretrained_dict.keys()):
            count2 = count2 + 1.0
            model_dict[k] = pretrained_dict.get(('module.' + k))
    model.load_state_dict(model_dict)
    print("load " + str(count2 * 100 / count) + " %")
    assert int(count2 * 100 / count) == 100, "model loading error"
    for _, child in model.named_children():
        for _, params in child.named_parameters():
            params.requires_grad = False
    print('model_load')
    print("Pretrain weights loaded")
    model.eval()

    pth2onnx(model, args.output_name)

    onnx_model = onnx.load(args.output_name)
    onnx.checker.check_model(onnx_model)

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = model(x)
    ort_session = onnxruntime.InferenceSession(args.output_name)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested on bs1 input with ONNXRuntime, and the result looks good!")
    x = torch.randn(16, 3, 224, 224, requires_grad=True)
    torch_out = model(x)
    ort_session = onnxruntime.InferenceSession(args.output_name)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested on bs16 input with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    main()