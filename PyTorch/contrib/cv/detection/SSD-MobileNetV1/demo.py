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
import os

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.data_preprocessing import PredictionTransform
from vision.ssd.config import mobilenetv1_ssd_config as config
import torch
if torch.__version__ >= '1.8':
    import torch_npu
import cv2

def get_raw_data():
    from urllib.request import urlretrieve
    current_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(current_path, './url.ini'), 'r') as _f:
        _content = _f.read()
        image_url = _content.split('image_url=')[1].split('\n')[0]
    urlretrieve(image_url, 'tmp.jpg')
    image = cv2.imread("tmp.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def test():
    loc = 'npu:0'
    loc_cpu = 'cpu'
    torch.npu.set_device(loc)
    class_names = [name.strip() for name in open("models/voc-model-labels.txt").readlines()]
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    print("begin to load net")
    net.load("models/mb1-ssd.pth")
    DEVICE = torch.device(loc)
    net = net.to(DEVICE)
    print("load net ok")
    net.eval()
    transform = PredictionTransform(config.image_size, config.image_mean,
                          config.image_std)
    image = get_raw_data()
    image = transform(image)
    images = image.unsqueeze(0)
    images = images.to(DEVICE)

    scores, boxes = net.forward(images)

    print(f"scores shape: {scores.shape}")
    print(f"boxes shape: {boxes.shape}")

if __name__ == "__main__":
    test()
