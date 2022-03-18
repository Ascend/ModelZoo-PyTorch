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

#include "parrots_cpp_helper.hpp"

void TINShiftForwardCUDAKernelLauncher(const DArrayLite input,
                                       const DArrayLite shift,
                                       DArrayLite output, cudaStream_t stream);

void TINShiftBackwardCUDAKernelLauncher(const DArrayLite grad_output,
                                        const DArrayLite shift,
                                        DArrayLite grad_input,
                                        cudaStream_t stream);

void tin_shift_forward_cuda(CudaContext &ctx, const SSElement &attr,
                            const OperatorBase::in_list_t &ins,
                            OperatorBase::out_list_t &outs) {
  const auto &input = ins[0];
  const auto &shift = ins[1];
  auto &output = outs[0];
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  TINShiftForwardCUDAKernelLauncher(input, shift, output, stream);
}

void tin_shift_backward_cuda(CudaContext &ctx, const SSElement &attr,
                             const OperatorBase::in_list_t &ins,
                             OperatorBase::out_list_t &outs) {
  const auto &grad_output = ins[0];
  const auto &shift = ins[1];
  auto &grad_input = outs[0];
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  TINShiftBackwardCUDAKernelLauncher(grad_output, shift, grad_input, stream);
}

PARROTS_EXTENSION_REGISTER(tin_shift_forward)
    .input(2)
    .output(1)
    .apply(tin_shift_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(tin_shift_backward)
    .input(2)
    .output(1)
    .apply(tin_shift_backward_cuda)
    .done();
