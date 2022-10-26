#!/usr/bin/env python
# Copyright 2020 Huawei Technologies Co., Ltd
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
import argparse
import sys
import os
import torch
import torch_npu
import torchlight
from torchlight import import_class
import torch.multiprocessing as mp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class(
        'processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class(
        'processor.demo_offline.DemoOffline')
    # endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    if p.arg.rt2:
        torch.npu.set_compile_mode(jit_compile=False)

        option = {}
        option["NPU_FUZZY_COMPILE_BLACKLIST"] = "AvgPoolV2Grad"
        torch.npu.set_option(option)

    devices = [p.arg.device] if isinstance(
        p.arg.device, int) else list(p.arg.device)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '59629'
    if len(devices) > 1 or "gpu" in p.arg.use_gpu_npu:
        mp.spawn(p.parallel_train, nprocs=len(devices))
    else:
        p.parallel_train(p.arg.device[0])
