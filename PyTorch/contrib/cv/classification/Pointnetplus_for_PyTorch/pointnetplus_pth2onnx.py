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

from collections import OrderedDict
import torch
import torch.onnx
import argparse
import sys
sys.path.append('./models/models')
import pointnet2_cls_ssg as pointnet2_cls
from pointnet2_utils import farthest_point_sample
from pointnet2_utils import sample_and_group


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('off_line_pred')
    parser.add_argument('--target_model', type=int, default=1,
                        required=True, help='target trans_models')
    parser.add_argument('--pth_dir', type=str, default='',
                        required=False, help='target trans_models')
    parser.add_argument('--batch_size', type=int, default=1,
                        required=False, help='batch size')
    return parser.parse_args()


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def model_convert(dir):
    experiment_dir = dir
    dummy_input = torch.randn(args.batch_size, 3, 1024)
    checkpoint = torch.load(str(experiment_dir) + '/best_model.pth',map_location = 'cpu') 
    checkpoint['model_state_dict'] = proc_node_module(checkpoint,'model_state_dict')
    model = pointnet2_cls.get_model_part1(normal_channel=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    npoint = 512
    radius = 0.2
    nsample = 32
    points = None
    test_input = dummy_input.permute(0, 2, 1)
    centroid = farthest_point_sample(test_input, npoint)
    new_xyz, new_points = sample_and_group(npoint, radius, nsample, test_input, points, centroid)
    new_points = new_points.permute(0, 3, 2, 1)
    input_names = ["xyz", "samp_points"]
    output_names = ["l1_xyz", "l1_point"]
    torch.onnx.export(model, (new_xyz, new_points),
                      "Pointnetplus_part1_bs{}.onnx".format(args.batch_size),
                      input_names=input_names, verbose=True, output_names=output_names, opset_version=11)


def model_convert2(dir):
    experiment_dir = dir
    dummy_xyz_input = torch.randn(args.batch_size, 3, 512)
    dummy_point_input = torch.randn(args.batch_size, 128, 512)
    checkpoint = torch.load(str(experiment_dir) + '/best_model.pth',map_location = 'cpu')
    checkpoint['model_state_dict'] = proc_node_module(checkpoint,'model_state_dict')
    model = pointnet2_cls.get_model_part2(normal_channel=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    npoint = 128
    radius = 0.4
    nsample = 64
    points = None
    test_input = dummy_xyz_input.permute(0, 2, 1)
    test_points = dummy_point_input.permute(0, 2, 1)
    centroid = farthest_point_sample(test_input, npoint)
    new_xyz, new_points = sample_and_group(npoint, radius, nsample, test_input, test_points, centroid)
    new_points = new_points.permute(0, 3, 2, 1)
    new_xyz = new_xyz.permute(0, 2, 1)
    input_names = ["l1_xyz", "l1_points"]
    output_names = ["class", "l3_point"]

    torch.onnx.export(model, (new_xyz, new_points),
                      "Pointnetplus_part2_bs{}.onnx".format(args.batch_size),
                      input_names=input_names, verbose=True, output_names=output_names, opset_version=11)


if __name__ == '__main__':
    args = parse_args()
    if(args.target_model == 1):
        model_convert(args.pth_dir)
    else:
        model_convert2(args.pth_dir)
