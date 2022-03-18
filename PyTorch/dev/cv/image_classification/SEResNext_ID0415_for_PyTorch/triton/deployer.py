#!/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#


import sys
import os
import torch
import argparse
import triton.deployer_lib as deployer_lib


def get_model_args(model_args):
    """ the arguments initialize_model will receive """
    parser = argparse.ArgumentParser()
    ## Required parameters by the model.
    parser.add_argument(
        "--config",
        default="resnet50",
        type=str,
        required=True,
        help="Network to deploy",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="The checkpoint of the model. "
    )
    parser.add_argument(
        "--batch_size", default=1000, type=int, help="Batch size for inference"
    )
    parser.add_argument(
        "--fp16", default=False, action="store_true", help="FP16 inference"
    )
    parser.add_argument(
        "--dump_perf_data",
        type=str,
        default=None,
        help="Directory to dump perf data sample for testing",
    )
    return parser.parse_args(model_args)


def initialize_model(args):
    """ return model, ready to trace """
    from image_classification.resnet import build_resnet

    model = build_resnet(args.config, "fanin", 1000, fused_se=False)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in state_dict.items()}
        )
    return model.half() if args.fp16 else model


def get_dataloader(args):
    """ return dataloader for inference """
    from image_classification.dataloaders import get_syntetic_loader

    def data_loader():
        loader, _ = get_syntetic_loader(None, 128, 1000, True, fp16=args.fp16)
        processed = 0
        for inp, _ in loader:
            yield inp
            processed += 1
            if processed > 10:
                break

    return data_loader()


if __name__ == "__main__":
    # don't touch this!
    deployer, model_argv = deployer_lib.create_deployer(
        sys.argv[1:]
    )  # deployer and returns removed deployer arguments

    model_args = get_model_args(model_argv)

    model = initialize_model(model_args)
    dataloader = get_dataloader(model_args)

    if model_args.dump_perf_data:
        input_0 = next(iter(dataloader))
        if model_args.fp16:
            input_0 = input_0.half()

        os.makedirs(model_args.dump_perf_data, exist_ok=True)
        input_0.detach().cpu().numpy()[0].tofile(
            os.path.join(model_args.dump_perf_data, "input__0")
        )

    deployer.deploy(dataloader, model)
