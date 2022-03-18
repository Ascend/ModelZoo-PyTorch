# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import torch
import torch.onnx
import sys
import ssl

from models import ICNet

def convert(pth_file, onnx_file):
    
    model = ICNet(nclass=19, backbone='resnet50',train_mode=False)
    #print(model)
    pretrained_net = torch.load(pth_file, map_location='cpu')
    model.load_state_dict(pretrained_net)
    model.eval()
    input_names = ["actual_input_1"]
    dummy_input = torch.randn(1, 3, 1024, 2048)
    
    torch.onnx.export(model, dummy_input, onnx_file, input_names=input_names, opset_version=11, verbose=True)

if __name__ == "__main__":

    # pth_file = sys.argv[1]
    # onnx_file = sys.argv[2]
    
    pth_file = 'rankid0_icnet_resnet50_192_0.687_best_model.pth'
    onnx_file = 'ICNet.onnx'

    ssl._create_default_https_context = ssl._create_unverified_context
    convert(pth_file, onnx_file)
