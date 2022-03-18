"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import torch
import torchvision
from torchvision import datasets, transforms
from collections import OrderedDict

from timm.models import create_model
from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor

DEFAULT_CROP_PCT = 0.875

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--checkpoint', type=str, default='',
                    help='checkpoint path')
args = parser.parse_args()

def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

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
    if args.checkpoint == '':
        print("please give the checkpoint path using --checkpoint param")
        exit(0)
    checkpoint = torch.load(args.checkpoint, map_location=loc)
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    
    model = create_model(
        'volo_d1',
        pretrained=False,
        num_classes=None,
        drop_rate=0.0,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=0.0,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        scriptable=False,
        checkpoint_path='',
        img_size=224)
    
    model = model.to(loc)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    crop_pct = DEFAULT_CROP_PCT
    img_size = 224
    scale_size = int(math.floor(img_size / crop_pct))
    interpolation = 'bilinear'
    tfl = [
        transforms.Resize(scale_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    tfl += [ToNumpy()]
    data_transfrom = transforms.Compose(tfl)

    rd = get_raw_data()

    inputs = data_transfrom(rd)
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