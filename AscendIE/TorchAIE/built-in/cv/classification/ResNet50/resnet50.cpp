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

#include <vector>
#include <string>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>

#include "torch_aie.h"

int main(int argc, const char* argv[])
{
    std::string modelPath = argv[1];
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(modelPath);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model from : " << modelPath << std::endl;
        return -1;
    }
    torch_aie::set_device(0);
    // 1.compile module
    std::vector<torch_aie::Input> inputs;
    std::vector<int64_t> inputShape = {64, 3, 224, 224};
    inputs.push_back(torch_aie::Input(inputShape, torch_aie::DataType::FLOAT, torch_aie::TensorFormat::NCHW));
    torch_aie::torchscript::CompileSpec compileInfo(inputs);
    auto compileModule = torch_aie::torchscript::compile(module, compileInfo);

    // 2.prepare data
    std::vector<torch::jit::IValue> inputIvalues;
    inputIvalues.push_back(at::randn(inputShape, torch::kFloat));

    // 3.forward
    auto aieRes = compileModule.forward(inputIvalues);
    std::cout << "aieRes:" << aieRes << std::endl;

    auto jitRes = module.forward(inputIvalues);
    std::cout << "jitRes:" << jitRes << std::endl;

    return 0;
}