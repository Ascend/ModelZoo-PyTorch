# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import argparse
from config import get_config
from classification import build_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp
from torchvision import transforms

def parse_option():
    parser = argparse.ArgumentParser('Focal Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', default='./configs/focalv2_small_useconv_patch4_window7_224.yaml', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', default=64, type=int, help="batch size for single GPU")
    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--debug', action='store_true', help='Perform debug only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=1, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def build_transform():
    t = []
    size = int((256 / 224) * 224)
    t.append(
        transforms.Resize(size, interpolation=_pil_interp('bicubic')),
    )
    t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def get_raw_data():
    from PIL import Image
    from urllib.request import urlretrieve
    IMAGE_URL = 'https://bbs-img.huaweicloud.com/blogs/img/thumb/1591951315139_8989_1363.png'
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img

def test():
    loc = 'npu:0'
    loc_cpu = 'cpu'
    torch.npu.set_device(loc)
    checkpoint = torch.load("./focalv2-small-useconv-is224-ws7.pth", map_location=loc)

    _, config = parse_option()
    model = build_model(config)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(loc)
    model.eval()

    rd = get_raw_data()
    preprocess = build_transform()
    inputs = preprocess(rd)

    inputs = inputs.unsqueeze(0)
    inputs = inputs.to(loc)
    output = model(inputs)
    output = output.to(loc_cpu)

    _, pred = output.topk(1, 1, True, True)
    result = torch.argmax(output, 1)
    print("class: ", pred[0][0].item())
    print(result)

if __name__ == "__main__":
    test()