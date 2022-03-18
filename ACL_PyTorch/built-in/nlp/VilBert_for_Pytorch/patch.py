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

import os
import sys


try:
    import allennlp
    module_path = os.path.dirname(allennlp.__file__)

    print("allennlp path:", module_path)
    os.system(f"cp -rf allennlp_patch/* {module_path}")
except:
    print("allennlp not found, please check")

try:
    import allennlp_models
    module_path = os.path.dirname(allennlp_models.__file__)

    print("allennlp_models path:", module_path)
    os.system(f"cp -rf allennlp_models_patch/* {module_path}")
except:
    print("allennlp_models not found, please check")
