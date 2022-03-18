# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import sys

sys.path.append('./VideoPose3D')
from common.model import TemporalModel
from collections import OrderedDict
import argparse
import torch
from torch.serialization import load

def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def vp3d_path2onnx(args):
    num_joints = 17
    joints_dim = 2
    filter_widths = [3,3,3,3,3]

    model_pos = TemporalModel(num_joints, joints_dim, num_joints, filter_widths=filter_widths,
        causal=False, dropout=0.25, channels=1024, dense=False)
    dummy_input = torch.randn(2, 6115, num_joints, joints_dim)
    chk_filename = args.model
    print(f'Loading checkpoint {chk_filename}')
    checkpoint = torch.load(chk_filename, map_location='cpu')
    checkpoint = proc_nodes_module(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint)

    output_file = args.onnx
    input_names = ['2d_poses']
    output_names = ['3d_preds']
    # dynamic_axes = {'2d_poses':{0:'2',1:'1024'},'3d_preds':{0:'2',1:'1024'}}

    model_pos.eval()
    torch.onnx.export(model_pos, dummy_input, output_file, input_names=input_names, output_names=output_names, 
                        opset_version=11, verbose=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="vp3d to onnx")
    parser.add_argument('-m', '--model', default='./checkpoint/model_best.bin', 
                        type=str, metavar='PATH', help="path to model")
    parser.add_argument('-o', '--onnx', default='vp3d.onnx')

    args = parser.parse_args()
    vp3d_path2onnx(args)
