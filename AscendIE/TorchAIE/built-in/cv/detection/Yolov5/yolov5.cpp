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

namespace {
const std::string MODULE_DIR = "yolov5s.torchscript";
const std::string TORCHAIE_MODULE_DIR = "yolov5sb1_torch_aie.torchscript";

bool almostEqual(const at::Tensor &computedTensor, const at::Tensor &gtTensor, float atol, float rtol)
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

auto getCompileSpec() -> torch_aie::torchscript::CompileSpec
{
    std::vector<int64_t> shapeOpt = { 1, 3, 640, 640 };
    std::vector<torch_aie::Input> inputs;
    inputs.emplace_back(torch_aie::Input(shapeOpt, torch_aie::DataType::FLOAT, torch_aie::TensorFormat::NCHW));
    torch_aie::torchscript::CompileSpec compileSpec(inputs);
    return std::move(compileSpec);
}

auto compileModule() -> torch::jit::Module
{
    // Load Module
    torch::jit::script::Module module = torch::jit::load(MODULE_DIR);
    module.eval();

    // set compile spec
    auto compile_spec = getCompileSpec();

    // torch_aie compile
    auto torchAieModule = torch_aie::torchscript::compile(module, compile_spec);
    std::cout << "compile done" << std::endl;
    return torchAieModule;
}

void saveModule(torch::jit::Module torchAieModule)
{
    torchAieModule.save(TORCHAIE_MODULE_DIR);
    std::cout << "torch_aie save done" << std::endl;
}

void PrepareInputData(std::vector<torch::jit::IValue> &inputIvalue, const std::vector<int64_t> &shape,
    const at::ScalarType &dtype)
{
    inputIvalue.push_back(at::randn(shape, dtype));
}

void forwardAndCompare()
{
    torch_aie::set_device(0);
    // load ts
    torch::jit::script::Module jitModule = torch::jit::load(MODULE_DIR);
    jitModule.eval();

    // load compiled model
    auto torchAieModule = torch::jit::load(TORCHAIE_MODULE_DIR);
    std::cout << "load torchAie model done!" << std::endl;

    // prepare data
    std::vector<torch::jit::IValue> inputIvalue;
    PrepareInputData(inputIvalue, { 1, 3, 640, 640 }, torch::kFloat);

    // foward and compare
    auto jitResults = jitModule.forward(inputIvalue).toTuple()->elements();
    auto aieResults = torchAieModule.forward(inputIvalue).toTuple()->elements();
    bool compareResult = true;
    std::cout << "aieResults size is: " << aieResults.size() << std::endl;
    std::cout << "jitResults size is: " << jitResults.size() << std::endl;

    std::cout << "aieResults[1].tagKind() is: " << aieResults[1].tagKind() << std::endl;
    std::cout << "jitResults[1].tagKind() is: " << jitResults[1].tagKind() << std::endl;
    std::cout << "aieResults[1] size is: " << aieResults[1].toList().size() << std::endl;
    std::cout << "jitResults[1] size is: " << jitResults[1].toList().size() << std::endl;

    bool compareTemp = almostEqual(aieResults[0].toTensor(), jitResults[0].toTensor(), 5e-3, 5e-3);
        compareResult = compareResult && compareTemp;
    for (int i = 0; i < aieResults[1].toList().size(); ++i) {
        compareTemp = almostEqual(aieResults[1].toList().get(i).toTensor(), jitResults[1].toList().get(i).toTensor(), 5e-3, 5e-3);
        compareResult = compareResult && compareTemp;
    }
    if (compareResult == true) {
        std::cout << "compare pass!" << std::endl;
    } else {
        std::cout << "compare failed!" << std::endl;
    }
}
} // namespace

int main()
{
    torch::jit::Module torchAieModule = compileModule();
    saveModule(torchAieModule);
    forwardAndCompare();
    return 0;
}