# ============================================================================
# Copyright 2018-2019 Open-MMLab. All rights reserved.
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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
from mmskeleton.utils import call_obj, load_checkpoint


def pt2onnx():
    """ST-GCN export script to convert pt model to ONNX model.
    Args:
        -ckpt: input checkpoint file path
        -onnx: output onnx file path
        -batch_size: define batch_size of the model

    Returns:
        Null. export onnx model with the input parameter -onnx
    """
    # define input parameter
    parser = argparse.ArgumentParser(
        description='ST-GCN Pytorch model convert to ONNX model')
    parser.add_argument('-ckpt', 
        default='./checkpoints/st_gcn.kinetics-6fa43f73.pth', 
        help='input checkpoint file path')
    parser.add_argument('-onnx', 
        default='./st-gcn_kinetics-skeleton_bs1.onnx', 
        help='output onnx file path')
    parser.add_argument('-batch_size', default=1, 
        help='define batch_size of the model')
    args = parser.parse_args()

    model_cfg = {'type': 'models.backbones.ST_GCN_18',
                'in_channels': 3,
                'num_class': 400,
                'edge_importance_weighting': True,
                'graph_cfg': {'layout': 'openpose', 'strategy': 'spatial'}}
    model = call_obj(**model_cfg)
    print("========= ST_GCN model ========")
    print(model)
    print("===============================")
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model.eval()

    input_name = ["input1"]
    output_name = ["output1"]
    dummy_input = torch.randn(int(args.batch_size), 3, 300, 18, 2, device='cpu')
    torch.onnx.export(model, dummy_input, args.onnx,
        input_names=input_name, output_names=output_name)

if __name__ == "__main__":
    pt2onnx()