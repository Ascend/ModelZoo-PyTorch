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
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from ..nn.squeezenet import squeezenet1_1

from .ssd import SSD
from .predictor import Predictor
from .config import squeezenet_ssd_config as config


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_squeezenet_ssd_lite(num_classes, is_test=False):
    base_net = squeezenet1_1(False).features  # disable dropout layer

    source_layer_indexes = [
        12
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2),
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_squeezenet_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor