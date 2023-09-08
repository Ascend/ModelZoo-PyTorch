# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Train"""
import torch
import torch_npu
import deepspeed_npu
from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain
from torch_npu.contrib import transfer_to_npu
from datetime import datetime
import os
if __name__ == "__main__":


    torch.npu.set_compile_mode(jit_compile=True)
    option = {"NPU_FUZZY_COMPILE_BLACKLIST": "Tril,SoftmaxV2,LayerNormGrad", "MM_BMM_ND_ENABLE": 'enable'}
    torch.npu.set_option(option)
    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined

    # os.environ["RANK_ID"] = str(neox_args.rank)
    from datetime import datetime
    time_str_result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    #     # 设置Profiling采集的Level，默认Level0（支持设置Level0、Level1和Level2）
    #     aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    #     # Level0默认不采集AIC Metrics，Level1和Level2默认开启Metrics，默认为PipeUtilization（支持 PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、ResourceConflictRatio、L2Cache）
    #     l2_cache=False  # 选择是否采集L2Cache的数据，耗时开关，默认False
    # )

    # torch.cuda.set_device(neox_args.rank)
    # with torch_npu.profiler.profile(
    #         activities=[
    #             torch_npu.profiler.ProfilerActivity.CPU,
    #             torch_npu.profiler.ProfilerActivity.NPU
    #         ],
    #         # schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=0, skip_first=0),
    #         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./../result/result_" + time_str_result[:len(time_str_result)-3]),
    #         experimental_config=experimental_config,
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True,
    #         with_flops=True,
    #         with_modules=True
    #         ) as prof:
    # with torch_npu.npu.profile(profiler_result_path="./../result/result_" + time_str_result[:len(time_str_result)-3], use_e2e_profiler=True):
    # print("当前进程：", os.getpid(), " 父进程：", os.getppid())

    # prof = torch_npu.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./result_dir"))
    # prof.__enter__()


    # pretrain(neox_args=neox_args, prof = prof)
    pretrain(neox_args=neox_args, prof=None)
    # prof.__exit__(None, None, None)



