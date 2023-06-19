#! -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Pretrain GPT"""
import os
import sys

sys.path.insert(0, './Megatron-DeepSpeed')
import deepspeed_npu
import torch
import torch_npu
import apex
from deepspeed_npu.adaptor_ops_adam_fused_adam import FusedAdamNPU
from torch.optim import AdamW

apex.optimizers.FusedAdam = FusedAdamNPU if 'FusedAdam' in os.environ else AdamW

import compression

sys.modules["deepspeed.compression"] = compression

from gpt_patch import gpt_patch
from pretrain_gpt import pretrain, train_valid_test_datasets_provider, forward_step, git_ds_info

if __name__ == "__main__":
    # Set jit_compile=True to eliminate TransData Op as the entire network is circulating in NZ Format.
    torch_npu.npu.set_compile_mode(jit_compile=True)
    git_ds_info()
    pretrain(train_valid_test_datasets_provider, gpt_patch.model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})