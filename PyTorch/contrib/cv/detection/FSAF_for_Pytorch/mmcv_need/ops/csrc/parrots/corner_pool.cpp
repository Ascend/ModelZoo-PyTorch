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

// Modified from
// https://github.com/princeton-vl/CornerNet-Lite/tree/master/core/models/py_utils/_cpools/src
#include "parrots_cpp_helper.hpp"

void bottom_pool_forward_cuda(CudaContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {}

void bottom_pool_backward_cuda(CudaContext& ctx, const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {}

void top_pool_forward_cuda(CudaContext& ctx, const SSElement& attr,
                           const OperatorBase::in_list_t& ins,
                           OperatorBase::out_list_t& outs) {}

void top_pool_backward_cuda(CudaContext& ctx, const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {}

void left_pool_forward_cuda(CudaContext& ctx, const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {}

void left_pool_backward_cuda(CudaContext& ctx, const SSElement& attr,
                             const OperatorBase::in_list_t& ins,
                             OperatorBase::out_list_t& outs) {}

void right_pool_forward_cuda(CudaContext& ctx, const SSElement& attr,
                             const OperatorBase::in_list_t& ins,
                             OperatorBase::out_list_t& outs) {}

void right_pool_backward_cuda(CudaContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {}

PARROTS_EXTENSION_REGISTER(bottom_pool_forward)
    .input(1)
    .output(1)
    .apply(bottom_pool_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(bottom_pool_backward)
    .input(2)
    .output(1)
    .apply(bottom_pool_backward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(top_pool_forward)
    .input(1)
    .output(1)
    .apply(top_pool_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(top_pool_backward)
    .input(2)
    .output(1)
    .apply(top_pool_backward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(left_pool_forward)
    .input(1)
    .output(1)
    .apply(left_pool_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(left_pool_backward)
    .input(2)
    .output(1)
    .apply(left_pool_backward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(right_pool_forward)
    .input(1)
    .output(1)
    .apply(right_pool_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(right_pool_backward)
    .input(2)
    .output(1)
    .apply(right_pool_backward_cuda)
    .done();
