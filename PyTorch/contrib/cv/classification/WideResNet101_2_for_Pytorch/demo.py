# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.npu
import models.resnet_0_6_0 as resnet_0_6_0

from url_utils import get_url

loc = 'npu:0'


def build_model():
    # 请自定义模型并加载预训练模型
    print("=> using pre-trained model wide_resnet101_2")
    model = resnet_0_6_0.wide_resnet101_2()
    print("loading model of yours...")
    pretrained_dict = torch.load("./model_best.pth.tar", map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_dict.items()})
    if "fc.weight" in pretrained_dict:
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()  # 注意设置eval模式
    return model


def get_raw_data():
    from PIL import Image
    from urllib.request import urlretrieve
    IMAGE_URL = get_url("image_url")
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img


def pre_process(rd):
    from torchvision import transforms
    transforms_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_data = transforms_list(rd)
    return input_data.unsqueeze(0)


def post_process(out):
    return torch.argmax(out, 1)


if __name__ == '__main__':
    torch.npu.set_device(loc)
    # 1. 获取原始数据
    raw_data = get_raw_data()

    # 2. 构建模型
    model_resnet = build_model()
    model_resnet = model_resnet.to(loc)

    # 3. 预处理
    input_tensor = pre_process(raw_data)
    input_tensor = input_tensor.to(loc)
    # 4. 执行forward
    output_tensor = model_resnet(input_tensor)
    output_tensor = output_tensor.cpu()
    # 5. 后处理
    result = post_process(output_tensor)
    # 6. 打印
    print(result)
