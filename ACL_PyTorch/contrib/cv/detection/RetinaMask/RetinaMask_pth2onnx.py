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
# limitations under the License

import os
import sys
import onnx
import torch
import argparse
import warnings

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")
warnings.filterwarnings("ignore")

from onnxsim import simplify
from collections import OrderedDict
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model


def pth2onnx():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str,
                        default="../configs/retina/retinanet_mask_R-50-FPN_2x_adjust_std011_ms.yaml")
    parser.add_argument("--weight_path", type=str, default="./npu_8P_model_0020001.pth")
    parser.add_argument("--save_path", type=str, default="./retinamask.onnx")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--simplify", type=bool, default=True)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg_path)
    cfg.freeze()

    onnx_file = args.save_path

    device = torch.device('cpu')
    model = build_detection_model(cfg)
    model = model.to(device)
    ckpt = torch.load(args.weight_path, map_location=device)
    checkpoints = ckpt['model']
    new_checkpoints = OrderedDict()
    for k, v in checkpoints.items():
        if k.startswith('module'):
            k = k[7:]
        new_checkpoints[k] = v
    model.load_state_dict(new_checkpoints)
    model.eval()

    dummy_input = torch.randn(args.batch_size, 3, 1344, 1344, dtype=torch.float32)
    # r = model(dummy_input)

    input_names = ["input"]
    output_names = ["bboxs", "labels", "scores", "masks"]

    torch.onnx.export(model,
                      dummy_input,
                      onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11,
                      verbose=False,
                      enable_onnx_checker=True)
    print("************* Convert to ONNX model file SUCCESS! *************")

    if args.simplify:
        sim_path = onnx_file
        onnx_model = onnx.load(onnx_file)
        onnx_sim_model, check = simplify(onnx_model, check_n=3)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_sim_model, sim_path)
        print('ONNX file simplified!')


if __name__ == '__main__':
    pth2onnx()
