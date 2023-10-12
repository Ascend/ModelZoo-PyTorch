/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include "core/util/jit_util.h"
#include "core/conversion/converters/converters.h"

namespace my_custom_converters {
using torch_aie::core::conversion::ConversionCtx;
using torch_aie::core::conversion::converters::args;
using torch_aie::core::conversion::Var;

auto g_cumsumRegstration = torch_aie::core::conversion::converters::RegisterNodeConverters()
    .pattern("aten::cumsum(Tensor self, int dim, *, int? dtype=None) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto inTensor = args[0].getFreezeAscendTensor(ctx);
            auto inputDims = inTensor->GetDimensions();
            auto dimIndex = args[1].unwrapToInt();
            int64_t inputDimsSize = inputDims.Size();
            if (dimIndex - inputDimsSize >= 0 || (inputDimsSize + dimIndex < 0)) {
                std::cerr << "Dimension out of range (expected to be inTensor range of [" << -inputDimsSize
                    << ", " << (inputDimsSize - 1) << "], but got " << dimIndex << ")";
            }
            if (dimIndex < 0) {
                dimIndex += inputDimsSize;
            }
            
            auto cumsumLayer = ctx->net->AddCumsum(inTensor, dimIndex);
            if (!cumsumLayer) {
                std::cerr << "[aten::cumsum] Unable to create cumsum layer for node: " << *n << std::endl;
            }

            cumsumLayer->SetName(torch_aie::core::util::nodeInfo(n).c_str());
            ctx->AssociateValueAndVar(n->outputs()[0], Var(cumsumLayer->GetOutput(0)));
            std::cout << "[aten::cumsum] node convert successfully" << std::endl;
            return true;
        });

} // namespace my_custom_converters