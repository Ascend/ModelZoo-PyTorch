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
from .agc import adaptive_clip_grad
from .checkpoint_saver import CheckpointSaver
from .clip_grad import dispatch_clip_grad
from .cuda import ApexScaler, NativeScaler
from .distributed import distribute_bn, reduce_tensor
from .jit import set_jit_legacy
from .log import setup_default_logging, FormatterNoInfo
from .metrics import AverageMeter, accuracy
from .misc import natural_key, add_bool_arg
from .model import unwrap_model, get_state_dict
from .model_ema import ModelEma, ModelEmaV2
from .summary import update_summary, get_outdir
