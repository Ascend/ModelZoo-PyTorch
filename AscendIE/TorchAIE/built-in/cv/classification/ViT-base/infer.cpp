/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>
#include "torch/csrc/jit/api/module.h"

#include "torch_aie.h"

using namespace std;

namespace {
    bool AlmostEqual(const at::Tensor &computedTensor, const at::Tensor &gtTensor,
                     float atol, float rtol = 1e-5)
    {
        auto computedTensorFloat = computedTensor.toType(at::kFloat);
        auto gtTensorFloat = gtTensor.toType(at::kFloat);

        auto diff = computedTensorFloat - gtTensorFloat;
        auto result = diff.abs().max().item<float>();
        auto threshold = atol + (rtol * gtTensor.abs().max().item<float>());

        std::cout << "Max Difference:" << std::to_string(result) << std::endl;
        std::cout << "Acceptable Threshold:" << std::to_string(threshold) << std::endl;

        return result <= threshold;
    }
}

int main(int argc, const char* argv[])
{
    const std::string modelPath = argv[1];
    const int imageSize = atoi(argv[2]);
    const int batchSize = atoi(argv[3]);
    const int numChannels = 3;

    torch_aie::set_device(0);
    torch::jit::script::Module module = torch::jit::load(modelPath);
    module.eval();

    // 1.compile module
    std::cout << "Start compiling module..." << std::endl;
    std::vector<int64_t> inputShape = {batchSize, numChannels, imageSize, imageSize};
    std::vector<torch_aie::Input> inputs;
    inputs.push_back(torch_aie::Input(inputShape, torch_aie::DataType::FLOAT, torch_aie::TensorFormat::NCHW));
    torch_aie::torchscript::CompileSpec compile_spec(inputs);
    compile_spec.precision_policy = torch_aie::PrecisionPolicy::FP16;
    compile_spec.allow_tensor_replace_int = true;
    auto compiledModule = torch_aie::torchscript::compile(module, compile_spec);
    std::cout << "Finish compiling module..." << std::endl;

    // 2 prepare data
    std::vector<torch::jit::IValue> inputIvalues;
    inputIvalues.push_back(at::randn(inputShape, torch::kFloat));

    // 3 forward
    auto aieResult = compiledModule.forward(inputIvalues);
    auto jitResult = module.forward(inputIvalues);
    std::cout << "AIE result:" << aieResult << std::endl;
    std::cout << "JIT result:" << jitResult << std::endl;

    bool compResult = AlmostEqual(aieResult.toTensor(), jitResult.toTensor(), 1e-2);
    if (compResult) {
        cout << "[SUCCESS] AIE inference result is the same as JIT!" << endl;
    } else {
        cout << "[Failure] AIE inference result is different from JIT!" << endl;
    }

    return 0;
}
