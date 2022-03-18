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
import sys
import torch
sys.path.append('./RefineDet.PyTorch')
from models.refinedet import build_refinedet
from data import VOCAnnotationTransform, VOCDetection, BaseTransform
from data import voc_refinedet
dataset_mean = (104, 117, 123)
cfg = voc_refinedet['320']



def pth2onnx(input_file, output_file, dataset):

    net = build_refinedet('test', 320, 21)

    net.load_state_dict(torch.load(input_file, map_location='cpu'))

    net.eval()
    input_names = ["image"]
    output_names = ['arm_loc_data', 'arm_conf_data', 
                'odm_loc_data', 'odm_conf_data', 'prior_data']
    dynamic_axes = {'image':{0:'-1'}, 
                    'arm_loc_data':{0:'-1'},
                    'arm_conf_data':{0:'-1'},
                    'odm_loc_data':{0:'-1'},
                    'odm_conf_data':{0:'-1'},
                    'prior_data':{0:'-1'}
                    }

    dummy_input, _, _, _ = dataset.pull_item(0)
    dummy_input = dummy_input.unsqueeze(0)
    

    torch.onnx.export(net, dummy_input, output_file, input_names=input_names,
         output_names=output_names, dynamic_axes = dynamic_axes, 
         opset_version=11, verbose=True)
    

if __name__ == '__main__':
    input_file, output_file, dataset_path = sys.argv[1:4]
    dataset = VOCDetection(root=dataset_path,
                           image_sets=[('2007', 'test')],
                           transform=BaseTransform(320, dataset_mean),
                           target_transform=VOCAnnotationTransform(),
                           dataset_name='VOC07test')
    pth2onnx(input_file, output_file, dataset)
