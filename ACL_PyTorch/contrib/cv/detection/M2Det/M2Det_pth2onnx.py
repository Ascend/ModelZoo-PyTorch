'''
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
'''
import sys
sys.path.insert(0, './M2Det')
import torch
import torch.onnx
from collections import OrderedDict
import argparse
print(sys.path)
from m2det import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from configs.CC import Config
from utils.core import init_net



parser = argparse.ArgumentParser(description='pth2onnx')
parser.add_argument('--c', '--config', default='M2Det/configs/m2det512_vgg16.py')
parser.add_argument('--d', '--dataset', default='COCO', help='VOC or COCO dataset')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--pth', '--pth_path', default='weights/m2det512_vgg.pth')
parser.add_argument('--onnx', '--onnx_path', default='m2det512.onnx')
Args = parser.parse_args()

def proc_nodes_module(checkpoint):
    '''
    Args:
        checkpoint: Network parameters.
    Returns:
        Create a new dictionary, remove the unnecessary key value "module"
    '''
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(args, cfg):
    '''
    Args:
        args.pth_path: Weight file path
        args.onnx_path: onnx file path
        cfg: configs
    '''
    print('pth:{}'.format(args.pth_path))
    print('onnx:{}'.format(args.onnx_path))
    net = build_net('test', 
                    size = cfg.model.input_size, # Only 320, 512, 704 and 800 are supported
                    config = cfg.model.m2det_config)
    init_net(net, cfg, args.resume_net)
    model = net
    
    checkpoint = torch.load(args.pth_path, map_location='cpu')
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)

    model.eval()
    input_names = ["image"]
    output_names = ["scores", "boxes"]
    #dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dynamic_axes = {'image':{0:'-1'}, 'scores':{0:'-1'}, 'boxes':{0:'-1'}}
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, dummy_input, args.onnx_path, input_names = input_names, 
                      dynamic_axes = dynamic_axes, output_names = output_names, 
                      verbose=True, opset_version=11)

if __name__ == "__main__":
    Cfg = Config.fromfile(Args.config)
    convert(Args, Cfg)