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

from __future__ import print_function
import os
import argparse
import torch
import sys

sys.path.append("./Pytorch_Retinaface")
from models.retinaface import RetinaFace
from data import cfg_mnet


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print("Missing keys:{}".format(len(missing_keys)))
    print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    print("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print("Loading pretrained model from {}".format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def trace_model(args):
    torch.set_grad_enabled(False)
    cfg = cfg_mnet
    cfg["pretrain"] = False
    model = RetinaFace(cfg=cfg, phase="test")
    model = load_model(model, args.trained_model, args.cpu)
    model.eval()
    print("Finished loading model!")
    print(model)
    device = torch.device("cpu" if args.cpu else "cuda")
    model = model.to(device)
    inputs = torch.randn(1, 3, args.long_side, args.long_side).to(device)
    ts_model = torch.jit.trace(model, inputs)
    ts_model.save(args.output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trace model")
    parser.add_argument(
        "-m",
        "--trained_model",
        default="./mobilenet0.25_Final.pth",
        type=str,
        help="Trained state_dict file path to open",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        default="retinaface.ts",
        type=str,
        help="torch script file name",
    )

    parser.add_argument(
        "--long_side",
        default=1000,
        help="when origin_size is false, long_side is scaled size(320 or 640 for long side)",
    )
    parser.add_argument(
        "--cpu", action="store_true", default=True, help="Use cpu inference"
    )

    args = parser.parse_args()
    trace_model(args)
