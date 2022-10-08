# Copyright 2022 Huawei Technologies Co., Ltd
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


import torch
import torch.onnx

from segm.model.factory import load_model

def pth2onnx(ckpt_path, onnx_path):

    device = torch.device("cpu")
    model, variant = load_model(ckpt_path)
    model.eval()
    dummy_input = torch.randn(1, 3, 768, 768)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names = ["input"],
        output_names = ["output"],
        opset_version=11,
        verbose=False,
        strip_doc_string=True,
        dynamic_axes={"input": {0: "-1"}, "output": {0: "-1"}}
    )


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('pytorch model convert to onnx.')
    parser.add_argument('-c', '--checkpoint-path', type=str, required=True,
        help='path to checkpoint file.')
    parser.add_argument('-o', '--onnx-path', type=str, required=True,
        help='path to save onnx model.')
    args = parser.parse_args()

    pth2onnx(args.checkpoint_path, args.onnx_path)

