"""
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
"""

import torch
import sys
import torch.nn as nn
import torchvision.models as models


class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""

class ResNetSimCLR(nn.Module):
    """ Simclr model """
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        """forward """
        return self.backbone(x)


def pth2onnx(input_file, output_file):
    """pth to onnx"""
    checkpoint = torch.load(input_file, map_location='cpu')
    model = ResNetSimCLR(base_model='resnet18', out_dim=128)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    input_name = ["input"]
    output_name = ["output"]

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy_input, output_file, input_names=input_name, output_names=output_name, verbose=True)


if __name__ == "__main__":
    input_pth = sys.argv[1]
    output = sys.argv[2]
    pth2onnx(input_pth, output)

