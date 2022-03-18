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


from .deprecated_fp16_utils import \
    DeprecatedFp16OptimizerHook as Fp16OptimizerHook
from .deprecated_fp16_utils import deprecated_auto_fp16 as auto_fp16
from .deprecated_fp16_utils import deprecated_force_fp32 as force_fp32
from .deprecated_fp16_utils import \
    deprecated_wrap_fp16_model as wrap_fp16_model

__all__ = ['auto_fp16', 'force_fp32', 'Fp16OptimizerHook', 'wrap_fp16_model']
