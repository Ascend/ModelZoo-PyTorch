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

import os
import sys
import argparse
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='enet_citys',
                    help='model name (default: enet_citys)')
parser.add_argument('--dataset', type=str, default='citys', choices=['pascal_voc, pascal_aug, ade20k, citys'],
                    help='dataset name (default: citys)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default='../datasets/citys/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--device', default='npu', type=str,
                        help='device npu/cpu')
args = parser.parse_args()


def demo(config):
    device = config.device
    if config.device == "npu":
        loc = 'npu:{}'.format(config.local_rank)
        torch.npu.set_device(loc)
        device = loc


    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(config.input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)

    model = get_model(args.model, local_rank=args.local_rank, pretrained=True, root=args.save_folder).to(device)
    print('Finished loading model!')

    model.eval()
    with torch.no_grad():
        output = model(images)

    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset)
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(args.outdir, outname))


if __name__ == '__main__':
    demo(args)
