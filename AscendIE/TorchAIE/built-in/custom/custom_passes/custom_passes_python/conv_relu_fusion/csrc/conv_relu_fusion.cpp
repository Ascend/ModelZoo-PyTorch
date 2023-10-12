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
#include "core/util/prelude.h"

namespace my_custom_passes {
using namespace torch_aie::core;
using namespace torch_aie::core::conversion::converters;
using namespace torch_aie::core::conversion;
constexpr size_t INPUT = 0;
constexpr size_t WEIGHT = 1;
constexpr size_t BIAS = 2;
constexpr size_t STRIDE = 3;
constexpr size_t PADDING = 4;
constexpr size_t DILATION = 5;
constexpr size_t GROUPS = 8;

class ConvReluFusion {
public:
    ConvReluFusion(ConversionCtx* ctx, const torch::jit::Node* n, args& args): ctx(ctx), n(n), layerArgs(args){}
    void AddConvReluLayer()
    {
        PreparePara();
        AddConvLayer();
        AddReluLayer();
        // Associate the computed tensor(reluTensor) with output
        ctx->AssociateValueAndVar(n->outputs()[0], Var(reluTensor));
    }

    void PreparePara()
    {
        input = layerArgs[INPUT].getAscendTensor();
        stride = AscendIE::Dims(layerArgs[STRIDE].unwrapToIntVector());
        padding = AscendIE::Dims(layerArgs[PADDING].unwrapToIntVector());
        dilation = AscendIE::Dims(layerArgs[DILATION].unwrapToIntVector());
        groups = layerArgs[GROUPS].unwrapToInt();

        // Reshape the parameters to 2D if needed
        constexpr int64_t ONE_DIM = 1;
        if (stride.Size() == ONE_DIM) {
            stride = util::unsqueezeDims(stride, 1, 1);
        }
        if (padding.Size() == ONE_DIM) {
            padding = util::unsqueezeDims(padding, 1, 0);
        }
        if (dilation.Size() == ONE_DIM) {
            dilation = util::unsqueezeDims(dilation, 1, 1);
        }

        auto dims = input->GetDimensions();
        origDims = dims;
        // Make sure the tensor is 4-dimensional. Otherwise, AscendIE dose not support the tensor
        if (origDims.Size() < 4) {
            input = addPadding(ctx, n, input, 4);
            dims = input->GetDimensions();
        }

        weight = Weights(ctx, layerArgs[WEIGHT].unwrapToAtTensor());
        auto weightShape = util::toVec(weight.shape);
        // Same as above. Ensure that the tensor is a 4-dimensional tensor.
        if (weight.shape.Size() < 4) {
            for (size_t i = weight.shape.Size(); i < 4; i++) {
                weightShape.push_back(1);
            }
            weight.shape = util::toDims(weightShape);
        }
        kernel = AscendIE::Dims({weight.shape[2], weight.shape[3]});

        if (layerArgs[BIAS].getIValue().isTensor()) {
            bias = Weights(ctx, layerArgs[BIAS].unwrapToAtTensor());
        } else {
            bias = Weights();
        }
    }

    void AddConvLayer()
    {
        auto convLayer = ctx->net->AddConvolutionLayer(input, weight.shape[0], kernel, weight.data, bias.data);
        convLayer->SetStrides(stride);
        convLayer->SetPaddings(padding);
        convLayer->SetDilations(dilation);
        convLayer->SetGroupNum(groups);
        convLayer->SetName((util::nodeInfo(n) + " convolution layer").c_str());
        convTensor = convLayer->GetOutput(0);
        convTensor = addUnpadding(ctx, n, convTensor, origDims.Size());
    }

    void AddReluLayer()
    {
        auto reluLayer = ctx->net->AddActivationLayer(convTensor, AscendIE::ActivationKind::RELU);
        reluLayer->SetName((util::nodeInfo(n) + " relu layer").c_str());
        reluTensor = reluLayer->GetOutput(0);
    }

private:
    ConversionCtx* ctx;
    const torch::jit::Node* n;
    args &layerArgs;
    AscendIE::Tensor* input;
    AscendIE::Tensor* convTensor;
    AscendIE::Tensor* reluTensor;
    AscendIE::Dims stride;
    AscendIE::Dims padding;
    AscendIE::Dims dilation;
    AscendIE::Dims kernel;
    AscendIE::Dims origDims;
    int64_t groups;
    Weights weight;
    Weights bias;
};

// Subgraph string
const std::string g_convWithReluPatternGraph = R"IR(
    graph(%0 : Tensor, %1 : Tensor, %2 : Tensor, %3 : int[], %4 : int[], %5 : int[], %6 : bool, %7 : int[], %8 : int, %9 : bool, %10 : bool, %11 : bool, %12 : bool):
        %13 : Tensor = aten::_convolution(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12)
        %14 : Tensor = aten::relu(%13)
        return (%14))IR";

// Subgraph function
auto g_convReluFusionRegistrations = torch_aie::core::custom_pass::RegisterSubgraphConverters().pattern(
    g_convWithReluPatternGraph,
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
        ConvReluFusion convReluFusion(ctx, n, args);
        convReluFusion.AddConvReluLayer();        
        return true;
    }
);
} // my_custom_passes