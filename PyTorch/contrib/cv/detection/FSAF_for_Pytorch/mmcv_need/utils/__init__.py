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

# flake8: noqa
# Copyright (c) Open-MMLab. All rights reserved.
from .config import Config, ConfigDict, DictAction
from .misc import (check_prerequisites, concat_list, deprecated_api_warning,
                   import_modules_from_strings, is_list_of, is_seq_of, is_str,
                   is_tuple_of, iter_cast, list_cast, requires_executable,
                   requires_package, slice_list, tuple_cast)
from .path import (check_file_exist, fopen, is_filepath, mkdir_or_exist,
                   scandir, symlink)
from .progressbar import (ProgressBar, track_iter_progress,
                          track_parallel_progress, track_progress)
from .testing import (assert_attrs_equal, assert_dict_contains_subset,
                      assert_dict_has_keys, assert_is_norm_layer,
                      assert_keys_equal, assert_params_all_zeros)
from .timer import Timer, TimerError, check_time
from .version_utils import digit_version, get_git_hash

try:
    import torch
except ImportError:
    __all__ = [
        'Config', 'ConfigDict', 'DictAction', 'is_str', 'iter_cast',
        'list_cast', 'tuple_cast', 'is_seq_of', 'is_list_of', 'is_tuple_of',
        'slice_list', 'concat_list', 'check_prerequisites', 'requires_package',
        'requires_executable', 'is_filepath', 'fopen', 'check_file_exist',
        'mkdir_or_exist', 'symlink', 'scandir', 'ProgressBar',
        'track_progress', 'track_iter_progress', 'track_parallel_progress',
        'Timer', 'TimerError', 'check_time', 'deprecated_api_warning',
        'digit_version', 'get_git_hash', 'import_modules_from_strings',
        'assert_dict_contains_subset', 'assert_attrs_equal',
        'assert_dict_has_keys', 'assert_keys_equal'
    ]
else:
    from .env import collect_env
    from .logging import get_logger, print_log
    from .parrots_wrapper import (
        CUDA_HOME, TORCH_VERSION, BuildExtension, CppExtension, CUDAExtension,
        DataLoader, PoolDataLoader, SyncBatchNorm, _AdaptiveAvgPoolNd,
        _AdaptiveMaxPoolNd, _AvgPoolNd, _BatchNorm, _ConvNd,
        _ConvTransposeMixin, _InstanceNorm, _MaxPoolNd, get_build_config)
    from .parrots_jit import jit, skip_no_elena
    from .registry import Registry, build_from_cfg
    __all__ = [
        'Config', 'ConfigDict', 'DictAction', 'collect_env', 'get_logger',
        'print_log', 'is_str', 'iter_cast', 'list_cast', 'tuple_cast',
        'is_seq_of', 'is_list_of', 'is_tuple_of', 'slice_list', 'concat_list',
        'check_prerequisites', 'requires_package', 'requires_executable',
        'is_filepath', 'fopen', 'check_file_exist', 'mkdir_or_exist',
        'symlink', 'scandir', 'ProgressBar', 'track_progress',
        'track_iter_progress', 'track_parallel_progress', 'Registry',
        'build_from_cfg', 'Timer', 'TimerError', 'check_time', 'CUDA_HOME',
        'SyncBatchNorm', '_AdaptiveAvgPoolNd', '_AdaptiveMaxPoolNd',
        '_AvgPoolNd', '_BatchNorm', '_ConvNd', '_ConvTransposeMixin',
        '_InstanceNorm', '_MaxPoolNd', 'get_build_config', 'BuildExtension',
        'CppExtension', 'CUDAExtension', 'DataLoader', 'PoolDataLoader',
        'TORCH_VERSION', 'deprecated_api_warning', 'digit_version',
        'get_git_hash', 'import_modules_from_strings', 'jit', 'skip_no_elena',
        'assert_dict_contains_subset', 'assert_attrs_equal',
        'assert_dict_has_keys', 'assert_keys_equal', 'assert_is_norm_layer',
        'assert_params_all_zeros'
    ]
