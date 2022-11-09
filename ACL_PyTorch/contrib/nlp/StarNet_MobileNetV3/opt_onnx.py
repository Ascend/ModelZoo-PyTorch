# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import torch
import argparse
import numpy as np

from auto_optimizer import OnnxGraph


class GridSample(torch.nn.Module):
    def __init__(self):
        super(GridSample, self).__init__()
  
    def forward(self, img, grid):
        
        result = torch.nn.functional.grid_sample(img, grid, align_corners=True)
        
        return result
        

def create_grid_sample(tmp_onnx):
    grid_sample = GridSample()
    grid_sample.eval()

    input_names = ['img', 'grid']
    output_names = ['output']

    dynamic_axes = {'img': {0: '-1'}, 
                    'grid': {0: '-1'}, 
                    'output': {0: '-1'}}

    img = torch.randn(1, 3, 32, 100)
    grid = torch.randn(1, 32, 100, 2)

    torch.onnx.export(grid_sample, 
                      (img, grid), 
                      tmp_onnx, 
                      input_names = input_names, 
                      output_names = output_names, 
                      opset_version=16, 
                      verbose=False,
                      dynamic_axes = dynamic_axes)
    
    print("[info] Create grid_sample done.")


def insert_grid_sample(in_onnx, grid_sampel, out_onnx):
    model_graph = OnnxGraph.parse(in_onnx)
    grid_sample_graph = OnnxGraph.parse(grid_sampel)
    
    # 1. delet "Unsqueeze_Pad_Squeeze_Transpose" in RARE_Resnet34_vd_nogrid_sim
    model_graph.remove('Unsqueeze_8')
    model_graph.remove('Pad_0')
    model_graph.remove('Squeeze_9')
    model_graph.remove('Transpose_6')

    model_graph.update_map()

    # 2. insert grid_sample to RARE_Resnet34_vd_nogrid_sim
    grid_sample_node = grid_sample_graph['GridSample_0']
    grid_sample_out = grid_sample_graph['output']

    grid_sample_out.name = 'GridSample_0_out'

    grid_sample_node.inputs = ['transpose_3.tmp_0', 'transpose_4.tmp_0']
    grid_sample_node.outputs = [grid_sample_out.name]

    model_graph.nodes.append(grid_sample_node)
    model_graph.value_infos.append(grid_sample_out)

    model_graph['Transpose_7'].inputs[0] = grid_sample_out.name

    model_graph['Add_9'].inputs = []
    model_graph['Add_9'].outputs = []
    model_graph.remove('Add_9')

    model_graph.update_map()

    model_graph.save(out_onnx)
    os.remove(grid_sampel)
    
    print("[info] Insert grid_sample done.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="opt onnx")  # task process paramater
    parser.add_argument('--in_onnx', type=str)
    parser.add_argument('--out_onnx', type=str)
    
    args = parser.parse_args()
    
    grid_sampel = "grid_sampel.onnx"  
    create_grid_sample(grid_sampel)
    
    insert_grid_sample(args.in_onnx, grid_sampel, args.out_onnx)
    
    print("[info] Optimize onnx success. result onnx is: {}".format(args.out_onnx))
    