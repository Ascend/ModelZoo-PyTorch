# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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

from models.superpoint import SuperPoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert pth to onnx of SuperPoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--output_file', type=str, default='superpoint.onnx',
        help='path to save onnx file')
    parser.add_argument(
        '--image_size', type=int, nargs='+', default=[1600, 1200],
        help='Image size after resize')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=3,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')

    args = parser.parse_args()
    return args


def pth2onnx(cfg, opt):
    model = SuperPoint(cfg.get('superpoint', {})).eval()

    w, h = opt.image_size[0], opt.image_size[1]
    input_data = {'image': torch.randn(1, 1, h, w)}
    input_names = ['image']
    output_names = ['keypoints', 'scores', 'descriptors']

    dynamic_axes = {
        'keypoints': {0: "-1"},
        'scores': {0: "-1"},
        'descriptors': {1: "-1"}
    }
    torch.onnx.export(
        model,
        (input_data, {}),
        opt.output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
        opset_version=16,
    )


if __name__ == '__main__':
    option = parse_args()
    config = {
        'superpoint': {
            'nms_radius': option.nms_radius,
            'keypoint_threshold': option.keypoint_threshold,
            'max_keypoints': option.max_keypoints
        }
    }
    pth2onnx(config, option)
