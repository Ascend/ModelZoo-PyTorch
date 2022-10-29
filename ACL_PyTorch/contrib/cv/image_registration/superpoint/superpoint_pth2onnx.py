# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from models.SuperPointNet_gauss2 import SuperPointNet_gauss2
import argparse
import torch
import torch.onnx
import torchvision.models as models
from collections import OrderedDict
from models.model_wrap import SuperPointFrontend_torch

parser = argparse.ArgumentParser(description='superpoint')
parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                    help='model path')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
#parser.add_argument('--model-pah', default='', type=str, metavar='PATH',
                    #help='model path')
args = parser.parse_args()

def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def pth2onnx():
    checkpoint = torch.load(args.model_path, map_location='cpu')
    #print(checkpoint)
    checkpoint['model_state_dict'] = proc_node_module(checkpoint, 'model_state_dict')
    #model = SuperPointFrontend_torch(weights_path = "./superPointNet_170000_checkpoint.pth.tar")  
    # 调整模型为eval mode
    model = SuperPointNet_gauss2()
    model.load_state_dict((checkpoint['model_state_dict']), False)
    model.eval()  
    # 输入节点名
    input_names = ["image"]  
    # 输出节点名
    output_names = ["class"]  
    dynamic_axes = {'image': {0: '1'}, 'class': {0: '-1'}} 
    dummy_input = torch.randn(args.batch_size, 1, 240, 320)
    #dummy_input = torch.randn(64, 1, 3, 3)
    filename = "sp-" + str(args.batch_size) + ".onnx"
    torch.onnx.export(model, dummy_input, filename, input_names = input_names, 
    dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True) 

if __name__ == "__main__":
    pth2onnx()
