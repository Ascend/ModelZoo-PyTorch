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
import torch
from utils.load_helper import load_pretrain
from utils.config_helper import load_config
import argparse


parser = argparse.ArgumentParser(description='pth2onnx')

parser.add_argument('--resume', default='SiamMask_VOT.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_vot.json',
                    help='hyperparameter of SiamRPN in json format')
parser.add_argument('--arch', dest='arch', default='Custom', choices=['Custom', ''],
                    help='architecture of pretrained model')
parser.add_argument('--output_dir', default='.', type=str)
parser.add_argument('-type',type=int,help='0:get_useful_pth， 1:pth2onnx')
args = parser.parse_args()

def get_useful_pth(output_dir, args):
    cfg = load_config(args)

    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(anchors=cfg['anchors'])
    else:
        model = models.__dict__[args.arch](anchors=cfg['anchors'])

    model = load_pretrain(model, args.resume)
    torch.save(model.features.state_dict(), f'{output_dir}/features.pth')
    torch.save(model.rpn_model.state_dict(), f'{output_dir}/rpn_model.pth')
    torch.save(model.mask_model.state_dict(), f'{output_dir}/mask_model.pth')
    torch.save(model.refine_model.state_dict(), f'{output_dir}/refine_model.pth')


def pth2onnx(output_dir, args):
    cfg = load_config(args)

    if args.arch == 'Custom':
        from custom import Custom, Refine
        model = Custom(anchors=cfg['anchors'])
        refine = Refine()
    else:
        model = models.__dict__[args.arch](anchors=cfg['anchors'])

    features = torch.load(f'{output_dir}/features.pth')
    rpn_model = torch.load(f'{output_dir}/rpn_model.pth')
    mask_model = torch.load(f'{output_dir}/mask_model.pth')
    refine_model = torch.load(f'{output_dir}/refine_model.pth')

    model.features_t.load_state_dict(features)
    model.features_s.load_state_dict(features)
    model.rpn_model.load_state_dict(rpn_model)
    model.mask_model_feature.load_state_dict(mask_model)
    model.mask_model_head.load_state_dict(mask_model)
    refine.load_state_dict(refine_model)

    model.eval()
    mask_input_names = ["template", "search"]
    mask_output_names = ['score', 'delta', 'mask', 'f0', 'f1', 'f2', 'corr_feature']
    mask_dynamic_axes = {
        'template': {0: 'batch_size'},
        'search': {0: 'batch_size'},
        'score': {0: 'batch_size'},
        'delta': {0: 'batch_size'},
        'mask': {0: 'batch_size'},
        'f0': {0: 'batch_size'},
        'f1': {0: 'batch_size'},
        'f2': {0: 'batch_size'},
        'corr_feature': {0: 'batch_size'}
    }

    refine_input_names = ['p0', 'p1', 'p2', 'p3']
    refine_output_names = ['mask']

    x = (
        torch.randn(1, 3, 127, 127),
        torch.randn(1, 3, 255, 255)
    )
    y = (
        torch.randn(1, 64, 61, 61),
        torch.randn(1, 256, 31, 31),
        torch.randn(1, 512, 15, 15),
        torch.randn(1, 256, 1, 1),
    )
    refine_dynamic_axes = {
        'p0': {0: 'batch_size'},
        'p1': {0: 'batch_size'},
        'p2': {0: 'batch_size'},
        'p3': {0: 'batch_size'},
        'mask': {0: 'batch_size'}
    }

    torch.onnx.export(model, x, f'{output_dir}/mask.onnx', input_names=mask_input_names, dynamic_axes=mask_dynamic_axes,
                      output_names=mask_output_names, opset_version=11, verbose=True)
    torch.onnx.export(refine, y, f'{output_dir}/refine.onnx', input_names=refine_input_names, dynamic_axes=refine_dynamic_axes,
                      output_names=refine_output_names, opset_version=11, verbose=True)

    print("done")


if __name__ == '__main__':
    if args.type == 0:
        get_useful_pth(args.output_dir, args)
    else:
        pth2onnx(args.output_dir, args)
