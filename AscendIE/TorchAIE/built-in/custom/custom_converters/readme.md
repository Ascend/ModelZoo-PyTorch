# 自定义converter
Torch-AIE中存在部分Pytorch算子没有支持NPU计算，或NPU计算效果不佳。出于这些原因，我们支持用户自己注册算子的NPU计算(注册converter)。

## 通过C++编写自定义converter
为了注册一个算子的NPU实现，我们需要将Pytorch算子的schema和实现组网操作的lambda函数，通过RegisterNodeConverters()
    .pattern()方法绑定。以cumsum算子为例，源码如下：

```c++
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
} // my_custom_converters
```

## 在Python中使用自定义converter
用setup.py脚本将编写好的自定义converter的c++文件编译生成.so文件，而后在python推理脚本中加载该动态库即可。

### 通过setup.py脚本生成动态链接库
为了在Python程序中能使用编写好的自定义converter，我们建议使用Pytorch提供的CppExtension生成.so库以便我们在Python程序中加载。如下是将cumsum算子的自定义converter打包到一个.so文件的python脚本：
```python
import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

# library_dirs should point to the libtorch_aie.so and the libascendie.so
# include_dirs should point to the dir that include the headers
TORCH_AIE_PATH = "/opt/buildtools/Python-3.9.11/lib/python3.9/site-packages/torch_aie"
ASCEND_IE_PATH = "/usr/local/Ascend/aie/latest"

ext_modules = [
    cpp_extension.CppExtension('custom_converter', ['./csrc/cumsum_converter.cpp'],
                                library_dirs=[(TORCH_AIE_PATH + "/lib/")],
                                libraries=["torch_aie"],
                                include_dirs=[
                                    TORCH_AIE_PATH + "/include/",
                                    ASCEND_IE_PATH + "/include/"
                                ])
]

setup(
    name='custom_converter',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
```
其中，TORCH_AIE_PATH需要对应实际Torch-AIE的whl安装路径，可以通过pip show torch_aie获取

通过如下命令运行这个python脚本：
```shell
python3 setup.py install --user
```
执行该命令并等待脚本执行完毕后，您能发现生成了几个新的文件夹，在这些文件夹中，您可以找到生成的.so库，该库可以直接加载到我们的Python程序中。

### 在Python程序中加载.so库
我们在python程序中使用torch.ops.load_library方法加载生成的.so库，从而使得Torch-AIE支持新编写的算子converter，示例如下：
```python
import torch
import torch_aie

# After "python3 setup.py install --user", you should find this .so file under generated "build" directory
torch.ops.load_library(
    "./build/lib.linux-x86_64-3.9/custom_converter.cpython-39-x86_64-linux-gnu.so"
)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        x = torch.cumsum(x, 0)
        return x

model = MyModel().eval()
inputs = torch.randint(1, 10, (5, 5 )).type(torch.float)
tracedModel = torch.jit.trace(model, inputs)

input = [torch_aie.Input((5, 5))]

torch_aie.set_device(0)

compiled_module = torch_aie.compile(
    tracedModel,
    inputs=input,
    precision_policy=torch_aie.PrecisionPolicy.FP16,
    truncate_long_and_double=True,
    require_full_compilation=False,
    allow_tensor_replace_int=False,
    min_block_size=1,
    torch_executed_ops=[],
    soc_version="Ascend310P3",
    optimization_level=0
)

inputs = torch.randint(1, 10, (5, 5)).type(torch.float)
results_aie = compiled_module.forward(inputs)
results_torch = tracedModel.forward(inputs)
print("input:\n", inputs)
print("aie output:\n", results_aie.to("cpu"))
print("torch output:\n", results_torch)
```

## 在C++中使用自定义converter
类似在Python中使用自定义converter，在C++中要使用自定义converter同样需要先生成一个.so文件，然后在C++推理脚本中动态链接该库。

>存在部分用户出于效率或精度等因素需要重写Torch-AIE已经开发好的converter，为了支持该能力，需要用户先将Torch-AIE的动态库和C++推理脚本编译出可执行程序，后动态加载包含用户自定义converter的动态库，以覆盖Torch-AIE自带的该算子的converter。

### 编写CMake文件并生成包含有自定义converter的动态库
将编写好的自定义converter文件编译链接上必要的库，打包成一个.so文件
```shell
PROJECT(CUSTOM_CONVERTERS)
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

add_compile_options(-fPIE -fstack-protector-all -fPIC -Wall -Wfatal-errors -O2)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(ASCEND_IE_ROOT "/usr/local/Ascend/torch_aie/latest" CACHE STRING "")
set(TORCH_ROOT "/usr/local/libtorch2.0.0" CACHE STRING "")
set(TORCH_AIE_ROOT "/usr/local/Ascend/aie/latest" CACHE STRING "")
set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

file(GLOB_RECURSE GLOB_SRC
    ${ROOT_DIR}/csrc/*.cpp
)

add_library(custom_converters SHARED  ${GLOB_SRC})
set_property(TARGET custom_converters PROPERTY CXX_STANDARD 14)

target_include_directories(custom_converters PUBLIC
    ${TORCH_ROOT}/include
    ${TORCH_ROOT}/include/torch/csrc/api/include
    ${ASCEND_IE_ROOT}/include
    ${TORCH_AIE_ROOT}/include
)

target_link_directories(custom_converters PUBLIC
    ${ASCEND_IE_ROOT}/lib
    ${TORCH_ROOT}/lib
    ${TORCH_AIE_ROOT}/lib
)

target_link_libraries(custom_converters PUBLIC
    ascendie
    c10
    torch
    torch_cpu
    torch_aie
)
```
需要确保ASCEND_IE_ROOT，TORCH_ROOT，TORCH_AIE_ROOT三个变量的值分别对应到环境中的推理引擎，torch，框架推理插件的路径。

### C++推理脚本中动态链接自定义converter库
在C++推理脚本中通过dlopen函数将已经打包出来的custom_converters.so动态载入
```c++
#include <iostream>
#include <vector>
#include <string>

#include <torch/torch.h>
#include <torch/script.h>
#include "torch/csrc/jit/api/module.h"

#include <dlfcn.h>
#include "torch_aie.h"

#define LIB_PATH "../../build/libcustom_converters.so"

int main()
{
    // Load custom converter
    auto customConverterLib = dlopen(LIB_PATH, RTLD_NOW);
    if (customConverterLib == NULL) {
        std::cerr << "[ERROR]Failed to load custom converter library." << std::endl;
    }

    // load model
    constexpr int DEVICE_ID = 0;
    torch_aie::set_device(DEVICE_ID);
    std::string modelPath = "./cumsum_model.pth";
    torch::jit::script::Module module = torch::jit::load(modelPath);
    module.eval();
    std::cout << "load module success." << std::endl;

    // set compile spec
    std::vector<int64_t> shapeOpt {5, 5};
    std::vector<torch_aie::Input> inputs;
    inputs.emplace_back(torch_aie::Input(shapeOpt, torch_aie::DataType::FLOAT, torch_aie::TensorFormat::ND));
    torch_aie::torchscript::CompileSpec compileSpec(inputs);
    compileSpec.soc_version = "Ascend310P3";
    std::cout << "set compile spec successfully." << std::endl;

    // compile module
    std::cout << "compile start." << std::endl;
    auto compiledModule = torch_aie::torchscript::compile(module, compileSpec);
    std::cout << "compile module successfully" << std::endl;

    // prepare data
    std::vector<torch::jit::IValue> inputsIValues;
    auto inTensor = at::randint(0, 10, shapeOpt, torch::kFloat);
    inputsIValues.push_back(inTensor);
    std::cout << "Prepare input data successfully." << std::endl;

    // forward
    auto aieResults = compiledModule.forward(inputsIValues);
    auto torchResults = module.forward(inputsIValues);
    std::cout << "compiledModule forward successfully." << std::endl;
    std::cout << "input Tensor is: \n" << inputsIValues[0] << std::endl;
    std::cout << "\naie output is: \n" << aieResults << std::endl;
    std::cout << "\ntorch output is: \n" << torchResults << std::endl;
    
    torch_aie::finalize();
    return 0;
}
```

### 编写CMake文件并编译C++推理脚本生成可执行程序
将C++推理脚本编译并链接上必要的pytorch库和Torch-AIE库生成可执行程序
```shell
PROJECT(MODEL_INFER)
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

add_compile_options(-fPIE -fstack-protector-all -fPIC -Wall -Wfatal-errors -O2)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(TORCH_ROOT "/usr/local/libtorch2.0.0" CACHE STRING "")
set(TORCH_AIE_ROOT "/usr/local/Ascend/aie/latest" CACHE STRING "")
set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

file(GLOB_RECURSE GLOB_SRC
    ${ROOT_DIR}/cumsum_model_infer.cpp
)

add_executable(cumsum_model_infer ${GLOB_SRC})
set_property(TARGET cumsum_model_infer PROPERTY CXX_STANDARD 14)

target_include_directories(cumsum_model_infer PUBLIC
    ${TORCH_ROOT}/include/
    ${TORCH_ROOT}/include/torch/csrc/api/include
    ${TORCH_AIE_ROOT}/include
)

target_link_directories(cumsum_model_infer PUBLIC
    ${TORCH_ROOT}/lib
    ${TORCH_AIE_ROOT}/lib
)

target_link_libraries(cumsum_model_infer PUBLIC
    c10
    torch
    torch_cpu
    torch_aie
    dl
)
```
需要注意，由于C++推理脚本中使用到了dlopen动态加载包含自定义converter的.so文件，CMake中target_link_libraries中需要链接上dl库