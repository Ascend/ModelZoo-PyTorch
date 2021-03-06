# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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

# Copyright (c) Open-MMLab. All rights reserved.
from .base_module import BaseModule
from .base_runner import BaseRunner
from .builder import RUNNERS, build_runner
from .checkpoint import (CheckpointLoader, _load_checkpoint,
                         _load_checkpoint_with_prefix, load_checkpoint,
                         load_state_dict, save_checkpoint, weights_to_cpu)
from .dist_utils import (allreduce_grads, allreduce_params, get_dist_info,
                         init_dist, master_only)
from .epoch_based_runner import EpochBasedRunner, Runner
from .fp16_utils import LossScaler, auto_fp16, force_fp32, wrap_fp16_model
from .hooks import (HOOKS, CheckpointHook, ClosureHook, DistSamplerSeedHook,
                    EMAHook, Fp16OptimizerHook, Hook, IterTimerHook,
                    LoggerHook, LrUpdaterHook, MlflowLoggerHook, OptimizerHook,
                    PaviLoggerHook, SyncBuffersHook, TensorboardLoggerHook,
                    TextLoggerHook, WandbLoggerHook)
from .iter_based_runner import IterBasedRunner, IterLoader
from .log_buffer import LogBuffer
from .optimizer import (OPTIMIZER_BUILDERS, OPTIMIZERS,
                        DefaultOptimizerConstructor, build_optimizer,
                        build_optimizer_constructor)
from .priority import Priority, get_priority
from .utils import get_host_info, get_time_str, obj_from_dict, set_random_seed

__all__ = [
    'BaseRunner', 'Runner', 'EpochBasedRunner', 'IterBasedRunner', 'LogBuffer',
    'HOOKS', 'Hook', 'CheckpointHook', 'ClosureHook', 'LrUpdaterHook',
    'OptimizerHook', 'IterTimerHook', 'DistSamplerSeedHook', 'LoggerHook',
    'PaviLoggerHook', 'TextLoggerHook', 'TensorboardLoggerHook',
    'WandbLoggerHook', 'MlflowLoggerHook', '_load_checkpoint',
    'load_state_dict', 'load_checkpoint', 'weights_to_cpu', 'save_checkpoint',
    'Priority', 'get_priority', 'get_host_info', 'get_time_str',
    'obj_from_dict', 'init_dist', 'get_dist_info', 'master_only',
    'OPTIMIZER_BUILDERS', 'OPTIMIZERS', 'DefaultOptimizerConstructor',
    'build_optimizer', 'build_optimizer_constructor', 'IterLoader',
    'set_random_seed', 'auto_fp16', 'force_fp32', 'wrap_fp16_model',
    'Fp16OptimizerHook', 'SyncBuffersHook', 'EMAHook', 'build_runner',
    'RUNNERS', 'allreduce_grads', 'allreduce_params', 'LossScaler',
    'CheckpointLoader', 'BaseModule', '_load_checkpoint_with_prefix'
]
