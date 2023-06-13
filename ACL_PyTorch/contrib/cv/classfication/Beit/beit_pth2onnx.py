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

import torch
import argparse
import os

from unilm.beit.modeling_finetune import beit_base_patch16_224
#============================================================================
# Functions
#============================================================================


def pth2onnx(input_file, output_file, batch_size):
    if os.path.exists(input_file):
        checkpoint = torch.load(input_file, map_location=torch.device('cpu'))
    else:
        print("download checkpoint from readme ...")

    model = beit_base_patch16_224(pretrained=False, num_classes=1000, drop_rate=0.0,
                                  drop_path_rate=0.1, attn_drop_rate=0.0, use_mean_pooling=True,
                                  init_scale=0.001, use_rel_pos_bias=True, use_abs_pos_emb=False,
                                  init_values=0.1,)

    model.load_state_dict(checkpoint['model'])
    model.eval()

    input_names = ['image']
    output_names = ['class']
    dynamic_axes = {'image': {0: '-1'}, 'class':{0: '-1'}}

    dummy_img = torch.randn(batch_size, 3, 224, 224)
    torch.onnx.export(model, dummy_img, output_file, dynamic_axes=dynamic_axes,
                      verbose=True, input_names=input_names, output_names=output_names, opset_version=11)

#============================================================================
# Main
#============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="beit_base_patch16_224_pt22k_ft22kto1k.pth")
    parser.add_argument('--target', type=str, default="beit_base_patch16_224.onnx")
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    pth2onnx(args.source, args.target, args.batch_size)


