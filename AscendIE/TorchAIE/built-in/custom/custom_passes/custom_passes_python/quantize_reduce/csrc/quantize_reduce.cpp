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

#include <torch/torch.h>
#include <torch/script.h>

#include "core/custom_pass/custom_pass.h"
#include "core/util/jit_util.h"

namespace my_custom_passes {
using torch_aie::core::conversion::ConversionCtx;
using torch_aie::core::conversion::converters::args;
using torch_aie::core::conversion::Var;
constexpr int QUANTIZE_PER_TENSOR_PARA_SIZE = 6;
constexpr int QUANTIZE_TENSOR = 0;
constexpr int SCALE = 1;
constexpr int OFFSET = 2;

// Subgraph string
const std::string g_quantizePatternGraph = R"IR(
    graph(%0 : Tensor, %1 : Tensor, %2 : Tensor, %3 : int, %x : int, %y : int):
        %4 : Tensor = aten::mul(%0, %1)
        %5 : Tensor = aten::add(%4, %2, %3)
        %6 : Tensor = aten::round(%5)
        %7 : Tensor = aten::clamp(%6, %x, %y)
        return (%7))IR";

// Subgraph function
auto g_quantizeRegistrations = torch_aie::core::custom_pass::RegisterSubgraphConverters().pattern(
    g_quantizePatternGraph,
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
        // Get subgraph input
        auto quantizeTensor = args[QUANTIZE_TENSOR].getFreezeAscendTensor(ctx);

        auto scaleTensor = args[SCALE].unwrapToAtTensor();
        auto tensorCPU = scaleTensor.to(at::kCPU).contiguous();
        double scale = *static_cast<double*>(tensorCPU.data_ptr());

        auto offsetTensor = args[OFFSET].unwrapToAtTensor();
        tensorCPU = offsetTensor.to(at::kCPU).contiguous();
        double offset = *static_cast<double*>(tensorCPU.data_ptr());

        // Pass parameters(quantize_tensor, scale and offset) into the quantization layer
        auto quantizeLayer = ctx->net->AddQuantize(quantizeTensor, scale);
        if (!(quantizeLayer)) {
            std::cerr << "Unable to create layer for quantize" << std::endl;
        }
        quantizeLayer->SetOffset(offset);
        
        quantizeLayer->SetName(torch_aie::core::util::nodeInfo(n).c_str()); // Set quantization Layer name
        auto quantizedTensor = quantizeLayer->GetOutput(0); // Get the quantized tensor
        // Associate the quantized tensor with output
        ctx->AssociateValueAndVar(n->outputs()[0], Var(quantizedTensor));

        return true;
    }
);
} // my_custom_passes