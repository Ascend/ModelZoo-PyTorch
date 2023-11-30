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

from models.superglue import SuperGlue


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert pth to onnx of SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--output_file', type=str, default='superglue.onnx',
        help='path to save onnx file')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--image_size', type=int, nargs='+', default=[1600, 1200],
        help='Image size after resize')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    args = parser.parse_args()
    return args


def pth2onnx(cfg, opt):
    model = SuperGlue(cfg.get('superglue', {})).eval()

    input_data = {
        'keypoints0': torch.randn(1, 40, 2),
        'scores0': torch.randn(1, 40),
        'descriptors0': torch.randn(1, 256, 40),
        'keypoints1': torch.randn(1, 40, 2),
        'scores1': torch.randn(1, 40),
        'descriptors1': torch.randn(1, 256, 40),
        'image0': torch.randn(1, 1, opt.image_size[1], opt.image_size[0]),
        'image1': torch.randn(1, 1, opt.image_size[1], opt.image_size[0])
    }
    input_names = [
        'keypoints0', 'scores0', 'descriptors0',
        'keypoints1', 'scores1', 'descriptors1',
    ]
    output_names = ['matches0', 'matches1', 'matching_scores0', 'matching_scores1']
    dynamic_axes = {
        'keypoints0': {1: "-1"},
        'scores0': {1: "-1"},
        'descriptors0': {2: "-1"},
        'keypoints1': {1: "-1"},
        'scores1': {1: "-1"},
        'descriptors1': {2: "-1"},
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
        'superglue': {
            'weights': option.superglue,
            'sinkhorn_iterations': option.sinkhorn_iterations,
            'match_threshold': option.match_threshold,
            'image_size': option.image_size,
        }
    }
    pth2onnx(config, option)
