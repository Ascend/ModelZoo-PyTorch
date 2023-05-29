# Copyright 2023 Huawei Technologies Co., Ltd
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
import argparse
from collections import OrderedDict

import torch

from det3d import torchie
from det3d.models import build_detector
import middle_conv


weight_mapping_rules = {
    "backbone.middle_conv.0": {
        "target": "backbone.middle_conv.middle_conv_0.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.1": {
        "target": "backbone.middle_conv.middle_conv_0.1.bn3d",
    },
    "backbone.middle_conv.3": {
        "target": "backbone.middle_conv.middle_conv_1.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.4": {
        "target": "backbone.middle_conv.middle_conv_1.1.bn3d",
    },
    "backbone.middle_conv.6": {
        "target": "backbone.middle_conv.middle_conv_2.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.7": {
        "target": "backbone.middle_conv.middle_conv_2.1.bn3d",
    },
    "backbone.middle_conv.9": {
        "target": "backbone.middle_conv.middle_conv_3.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.10": {
        "target": "backbone.middle_conv.middle_conv_3.1.bn3d",
    },
    "backbone.middle_conv.12": {
        "target": "backbone.middle_conv.middle_conv_4.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.13": {
        "target": "backbone.middle_conv.middle_conv_4.1.bn3d",
    },
    "backbone.middle_conv.15": {
        "target": "backbone.middle_conv.middle_conv_5.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.16": {
        "target": "backbone.middle_conv.middle_conv_5.1.bn3d",
    },
    "backbone.middle_conv.18": {
        "target": "backbone.middle_conv.middle_conv_6.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.19": {
        "target": "backbone.middle_conv.middle_conv_6.1.bn3d",
    },
    "backbone.middle_conv.21": {
        "target": "backbone.middle_conv.middle_conv_7.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.22": {
        "target": "backbone.middle_conv.middle_conv_7.1.bn3d",
    },
    "backbone.middle_conv.24": {
        "target": "backbone.middle_conv.middle_conv_8.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.25": {
        "target": "backbone.middle_conv.middle_conv_8.1.bn3d",
    },
    "backbone.middle_conv.27": {
        "target": "backbone.middle_conv.middle_conv_9.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.28": {
        "target": "backbone.middle_conv.middle_conv_9.1.bn3d",
    },
    "backbone.middle_conv.30": {
        "target": "backbone.middle_conv.middle_conv_10.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.31": {
        "target": "backbone.middle_conv.middle_conv_10.1.bn3d",
    },
    "backbone.middle_conv.33": {
        "target": "backbone.middle_conv.middle_conv_11.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.34": {
        "target": "backbone.middle_conv.middle_conv_11.1.bn3d",
    },
    "backbone.middle_conv.36": {
        "target": "backbone.middle_conv.middle_conv_12.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.37": {
        "target": "backbone.middle_conv.middle_conv_12.1.bn3d",
    },
    "backbone.middle_conv.39": {
        "target": "backbone.middle_conv.middle_conv_13.0.conv3d",
        "permute": True,
    },
    "backbone.middle_conv.40": {
        "target": "backbone.middle_conv.middle_conv_13.1.bn3d",
    },
}


def convert_weight_keys(_checkpoint: dict, attr_name: str) -> dict:
    rules = weight_mapping_rules
    new_state_dict = OrderedDict()
    for key, value in _checkpoint[attr_name].items():
        last_dot = key.rfind(".")
        name = key[:last_dot]
        attr = key[last_dot:]
        if name in rules:
            target_name = rules[name]["target"]
            if "permute" in rules[name]:
                # Convert weight from SpConv format to native format.
                # SpConv: [kernel_size(0 1 2), input_channel(3), output_channel(4)]
                # Native: [output_channel(4), input_channel(3), kernel_size(0 1 2)]
                value = value.permute(4, 3, 0, 1, 2)
            
            new_state_dict[target_name + attr] = value

        else:
            new_state_dict[key] = value

    return new_state_dict


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default='./SE-SSD/examples/second/configs/config.py', 
        help="test config file path"
    )
    parser.add_argument("--checkpoint", type=str, default='./se-ssd-model.pth',  help="checkpoint file")
    parser.add_argument("--save_path", type=str, default='./se-ssd.onnx',  help="Path to save onnx model.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging while exporting onnx.")
    return parser.parse_args()


if __name__ == "__main__":
    settings = parse_arguments()
    config = torchie.Config.fromfile(settings.config)

    model = build_detector(config.model, train_cfg=None, test_cfg=config.test_cfg)
    checkpoint = torch.load(settings.checkpoint, map_location="cpu")
    checkpoint["state_dict"] = convert_weight_keys(checkpoint, "state_dict")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    save_path = settings.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_data = {
        "voxels": torch.zeros(1, 20000, 5, 4),
        "coordinates": torch.zeros(1, 20000, 4),
        "num_points": torch.ones(1, 20000,),
    }

    dynamic_axes = {
        "voxels": {0: '-1'},
        "coordinates": {0: '-1'},
        "num_points": {0: '-1'},
    }

    torch.onnx.export(
        model,
        tuple(input_data.values()),
        save_path,
        dynamic_axes=dynamic_axes,
        verbose=settings.verbose,
        opset_version=11,
        input_names=("voxels", "coordinates", "num_points"),
        output_names=('box_preds', 'cls_preds', 'dir_cls_preds', 'iou_preds')
    )

    print("Done.")