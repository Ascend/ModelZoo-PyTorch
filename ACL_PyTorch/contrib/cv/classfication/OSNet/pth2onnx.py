# Copyright 2020 Huawei Technologies Co., Ltd
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


import torch
import torchreid
import sys


def pth2onnx(input_file, output_file):
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=751,
        loss='softmax',
        pretrained=False
    )
    
    checkpoint = torch.load(input_file, map_location = None)
    model.load_state_dict(checkpoint)
    model.eval()
    
    input_names = ['image']
    output_names = ['feature']
    dynamic_axes = {'image':{0:'-1'}, 'feature':{0:'-1'}}
    dummy_input = torch.randn(1,3,256,128)    # (batch, channel, height, width)
    torch.onnx.export(model, dummy_input, output_file,
                      input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)
    
 
def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    pth2onnx(input_file, output_file)
    print("Done")


if __name__ == '__main__':
    main()  