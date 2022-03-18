# Copyright 2021 Huawei Technologies Co., Ltd
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

g_feat_in = []
g_feat_out = []
g_grad_in = []
g_grad_out = []


def forward_hook_fn(module, input, output):
    g_feat_in.append(input)
    g_feat_out.append(output)
    print(module)
    print(input)
    print(output)


def backward_hook_fn(module, grad_input, grad_output):
    g_grad_in.append(grad_input)
    g_grad_out.append(grad_output)
    print(module)
    print(grad_input)
    print(grad_input)





