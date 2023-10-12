# 自定义pass

出于提高一些多算子子图场景的NPU计算速度，或计算精度等原因，Torch-AIE提供了自定义pass方法。通过该方法，用户可以注册相关子图对应的NPU计算逻辑。

## 通过C++编写自定义pass
以量化操作为例，我们可以注册一个含有aten::mul，aten::add，aten::round，aten::clamp这四个算子的子图的PASS，从而实现量化操作的自定义NPU计算，示例如下：

```c++
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
```

## 在Python中使用自定义pass

### 通过setup.py脚本生成动态链接库
为了在Python程序中能使用编写好的自定义pass，我们建议使用Pytorch提供的CppExtension生成.so库以便我们在Python程序中加载。如下是将量化的自定义pass打包到一个.so文件的python脚本：
```python
import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

# library_dirs should point to the libtorch_aie.so and the libascendie.so
# include_dirs should point to the dir that include the headers
TORCH_AIE_PATH = "/opt/buildtools/Python-3.9.11/lib/python3.9/site-packages/torch_aie"
ASCEND_IE_PATH = "/usr/local/Ascend/aie/latest"

ext_modules = [
    cpp_extension.CppExtension('custom_passes_reduce', ['./csrc/quantize_reduce.cpp'],
                                library_dirs=[(TORCH_AIE_PATH + "/lib/")],
                                libraries=["torch_aie"],
                                include_dirs=[
                                    TORCH_AIE_PATH + "/include/",
                                    ASCEND_IE_PATH + "/include/"
                                ])
]

setup(
    name='custom_passes_reduce',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
```
其中，TORCH_AIE_PATH需要对应实际Torch-AIE的whl安装路径，可以通过pip show torch_aie获取

通过如下命令运行这个python脚本：
```shell
python3 setup.py install --user
```
执行完该命令并等待脚本执行完毕后，您能发现生成了几个新的文件夹，在这些文件夹中，您可以找到生成的.so库，该库可以直接加载到我们的Python程序中。

### 在Python程序中加载.so库
我们在python程序中使用torch.ops.load_library方法加载生成的.so库，示例如下：
```python
import torch
import torch_aie

# After "python3 setup.py install --user", you should find this .so file under generated "build" directory
torch.ops.load_library(
    "./build/lib.linux-x86_64-3.9/custom_passes_reduce.cpython-39-x86_64-linux-gnu.so"
)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        x = x * 8.8 + 1.1
        x = torch.clamp(torch.round_(x), -128, 127)
        return x

model = MyModel().eval()
inputs = torch.randint(1, 10, (5, 5 )).type(torch.float)
tracedModel = torch.jit.trace(model, inputs)

compile_input = [torch_aie.Input((5, 5))]

torch_aie.set_device(0)

compiled_module = torch_aie.compile(
    tracedModel,
    inputs=compile_input,
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
results_torch = tracedModel.forward(inputs).type(torch.int8)
print("input:\n", inputs)
print("aie output:\n", results_aie.to("cpu"))
print("torch output:\n", results_torch)
```

## 在C++中使用自定义pass
C++程序中使用自定义pass有两种方式：
1. 类似python中的使用方式，首先生成一个.so文件，然后在C++推理脚本中动态链接该库；
2. 直接将编写的自定义pass的文件同C++程序以及torch_aie的库文件一起编译生成可执行二进制文件。
   
其中方式1可参考自定义converter部分readme中的示例，下面对方式2展开描述

### 编写CMake文件编译自定义pass和C++推理脚本并生成可执行程序
为了直接得到可执行文件，编写的CMakeLists.txt文件中将csrc目录下的.cpp文件（自定义pass实现部分）和C++推理脚本quantize_model_infer.cpp文件一同编译，并链接上必要的库，示例如下：
```shell
PROJECT(MODEL_INFER)
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

add_compile_options(-fPIE -fstack-protector-all -fPIC -Wall -Wfatal-errors -O2)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Configurable parameters
set(ASCEND_IE_ROOT "/usr/local/Ascend/torch_aie/latest" CACHE STRING "")
set(TORCH_ROOT "/usr/local/libtorch2.0.0" CACHE STRING "")
set(TORCH_AIE_ROOT "/usr/local/Ascend/aie/latest" CACHE STRING "")
set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

file(GLOB_RECURSE GLOB_SRC
    ${ROOT_DIR}/csrc/*.cpp
    ${ROOT_DIR}/quantize_model_infer.cpp
)

add_executable(quantize_model_infer ${GLOB_SRC})
set_property(TARGET quantize_model_infer PROPERTY CXX_STANDARD 14)

target_include_directories(quantize_model_infer PUBLIC
    ${TORCH_ROOT}/include/
    ${TORCH_ROOT}/include/torch/csrc/api/include
    ${ASCEND_IE_ROOT}/include/
    ${TORCH_AIE_ROOT}/include
)

target_link_directories(quantize_model_infer PUBLIC
    ${ASCEND_IE_ROOT}/lib
    ${TORCH_ROOT}/lib
    ${TORCH_AIE_ROOT}/lib
)

target_link_libraries(quantize_model_infer PUBLIC
    ascendie
    c10
    torch
    torch_cpu
    torch_aie
)
```
需要确保ASCEND_IE_ROOT，TORCH_ROOT，TORCH_AIE_ROOT三个变量的值分别对应到环境中的推理引擎，torch，框架推理插件的路径。

使用自定义pass的C++推理脚本和不使用自定义pass的推理脚本并无区别，无须修改。