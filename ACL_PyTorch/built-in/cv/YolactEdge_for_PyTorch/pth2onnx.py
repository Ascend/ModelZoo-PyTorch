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

import argparse
import os
import torch
from yolact_edge.data import cfg, set_cfg
from yolact_edge.yolact import Yolact
from yolact_edge.inference import str2bool, parse_args
from yolact_edge.utils.timer import enable


if __name__ == '__main__':

    global args
    args = parse_args()

    if args.config is not None:
        set_cfg(args.config)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        torch.set_default_tensor_type('torch.FloatTensor')

        x = torch.randn((1, 3, cfg.max_size, cfg.max_size))
        net = Yolact(training=False)
        net.load_weights(args.trained_model, args=args)
        net.eval()
        net.detect.use_fast_nms = args.fast_nms

        torch.onnx.export(net,
                            x,
                            'yolact_edge.onnx',
                            opset_version=11,
                            enable_onnx_checker=True)
