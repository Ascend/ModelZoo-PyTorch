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
# import torch_npu
# from main import useNPU

class DataPrefetcher:
    def __init__(self, loader,use_npu=False):
        self.loader = iter(loader)
        self.use_npu = use_npu
        if use_npu:
            self.stream = torch.npu.Stream()
        else:
            self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        if self.use_npu:
            with torch.npu.stream(self.stream):
                self.next_input = self.next_input.npu(non_blocking=True)
                self.next_target = self.next_target.npu(non_blocking=True)
        else:
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        if self.use_npu:
            torch.npu.current_stream().wait_stream(self.stream)
        else:
            torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.preload()
        return input, target
