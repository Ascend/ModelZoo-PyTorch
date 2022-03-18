 **介绍**

Ascend ModelZoo，欢迎各位开发者

 **贡献要求**

开发者提交的模型包括源码、readme、参考模型license文件、测试用例和readme，并遵循以下标准

请贡献者在提交代码之前签署CLA协议，“个人签署”，[链接](https://clasign.osinfra.cn/sign/Z2l0ZWUlMkZhc2NlbmQ=)

如您完成签署，可在自己提交的PR评论区输入/check-cla进行核实校验

 **一、源码**

1、训练及在线推理请使用python代码实现，Ascend平台离线推理请使用C++或python代码，符合第四部分编码规范

2、参考[sample](https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/Transformer_for_TensorFlow)

3、贡献者模型代码目录规则："modelzoo/contrib/框架/Research/应用领域(nlp、cv、audio等)/网络名_IDxxx_for_TensorFlow"（以tf为例，社区管理团队会在贡献完成进行整合）

4、从其他开源迁移的代码，请增加License声明

 **二、License规则**

* TensorFlow

    迁移场景

    1、迁移TensorFlow模型中若源项目已包含License文件则必须拷贝引用，否则在模型顶层目录下添加TensorFlow Apache 2.0 License [TensorFlow License链接](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)

    2、迁移TensorFlow框架开发的模型，需要在模型目录下每个源文件附上源社区TensorFlow Apache 2.0 License头部声明，并在其下追加新增完整华为公司License声明

    ```
    # Copyright 2017 The TensorFlow Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
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
    # ============================================================================
    # Copyright 2021 Huawei Technologies Co., Ltd
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
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
    ```
   开发场景

    1、基于TensorFlow框架开发模型，需在模型项目顶层目录下添加TensorFlow Apache 2.0 License [TensorFlow License链接](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)

    2、基于TensorFlow框架开发模型，需要在模型目录下每个源文件附上源社区华为公司Apache 2.0 License头部声明
    ```
    # Copyright 2021 Huawei Technologies Co., Ltd
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
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
    ```
* PyTorch

    迁移场景

    1、迁移PyTorch模型中若源项目录已包含PyTorch License文件则必须拷贝引用，否则在模型顶层目录下添加PyTorch BSD-3 License [PyTorch License链接](https://github.com/pytorch/examples/blob/master/LICENSE)

    2、迁移PyTorch第三方框架开发的模型，需要在模型目录下每个源文件附上源社区PyTorch BSD-3 License头部声明，并在其下追加新增一行华为公司License声明
    ```
    # BSD 3-Clause License
    #
    # Copyright (c) 2017 xxxx
    # All rights reserved.
    # Copyright 2021 Huawei Technologies Co., Ltd
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:
    #
    # * Redistributions of source code must retain the above copyright notice, this
    #   list of conditions and the following disclaimer.
    #
    # * Redistributions in binary form must reproduce the above copyright notice,
    #   this list of conditions and the following disclaimer in the documentation
    #   and/or other materials provided with the distribution.
    #
    # * Neither the name of the copyright holder nor the names of its
    #   contributors may be used to endorse or promote products derived from
    #   this software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    # ============================================================================
    ```

    开发场景

    1、基于PyTorch框架开发模型，需在模型项目下添加PyTorch BSD-3 License [PyTorch License链接](https://github.com/pytorch/examples/blob/master/LICENSE)

    2、基于PyTorch框架开发模型，需要在模型目录下每个源文件附上源社区华为公司Apache 2.0 License头部声明
     ```
    # Copyright 2021 Huawei Technologies Co., Ltd
    #
    # Licensed under the BSD 3-Clause License  (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # https://opensource.org/licenses/BSD-3-Clause
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
     ```

* MindSpore/ACL

    1、迁移或开发场景下MindSpore/ACL模型顶层目录下需要包含华为公司 License [华为公司 License链接](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)

    2、迁移或开发场景下MindSpore/ACL模型，需要在模型目录下每个源文件中添加区华为公司Apache 2.0 License头部声明
     ```
    # Copyright 2021 Huawei Technologies Co., Ltd
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
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
     ```

> 关于License声明时间，应注意： 2021年新建的文件，应该是Copyright 2021 Huawei Technologies Co., Ltd 2020年创建年份，2020年修改年份，应该是Copyright 2020 Huawei Technologies Co., Ltd

 **三、readme**

readme用于指导用户理解和部署样例，要包含如下内容：

- 简介：

1、模型的来源及原理；

2、模型复现的步骤，含训练、eval、在线/离线推理等，入口请封装成`.sh`、`.py`；

- 关键要求：

1、模型的出处、对数据的要求、免责声明等，开源代码文件修改需要增加版权说明；

2、模型转换得到的离线模型对输入数据的要求；

3、环境变量设置，依赖的第三方软件包和库，以及安装方法；

4、精度和性能达成要求：尽量达到原始模型水平；

5、预训练checkpoint、结果checkpoint请提供归档OBS、网盘链接，如来自开源需明确来源地址

6、数据集说明

- [ ] 不允许直接提供数据集的下载链接，可使用词汇：用户自行准备好数据集，可选用“XXX”，“XXX”，“XXX”

  例如：请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括ImageNet2012，CIFAR10、Flower等，包含train和val两部分。

- [ ] 脚本中不允许提供链接下载数据集，如果开源脚本上存在对应的链接，请修改或者删除对应的脚本

训练ReadMe写作可参考下面两个链接：

[Readme example1](https://gitee.com/ascend/modelzoo/blob/master/built-in/TensorFlow/Official/cv/image_classification/DenseNet121_ID0067_for_TensorFlow/README.md)

[Readme example2](https://www.hiascend.com/zh/software/modelzoo/detail/C/093ed0219cb14f068af33784c62cf7ec)

离线推理ReadMe写作可参考下面链接：

[Readme example1](https://gitee.com/ascend/modelzoo/tree/master/contrib/ACL_TensorFlow/Research/cv/AdvancedEAST_ID0130_for_ACL/README.md)

 **四、自测试用例**

提供模型的自测试用例和readme，提交PR需要门禁及模型测试用例通过，性能和精度检查通过

- 简介：

1、不同于完整的训练过程和全量数据集的推理，自测试用例的目的是验证提交代码基本功能可用，执行时长控制在10min之内(推理或训练只需执行有限的图片或step)；

2、提交PR中训练用例入口`train_testcase.sh`, 在线推理用例入口`online_inference_testcase.sh`, 离线推理用例入口`offline_inference_testcase.sh`；

3、提交PR后，会自动触发门禁流水，后台会根据用例入口shell，自动将代码分发到对应执行环境；

4、Jenkins预置账号：登录账号请联系华为工程师/接口人获取，登录之后，可以查看到用例执行日志

5、如果提交失败，请查看日志，修复代码或其他问题后，在你当前的PR中，评论“compile”即可重新触发用例执行

- 关键要求：

1、自测试用例命名严格按照上述简介2要求来书写，否则门禁会校验失败；

2、用例应当包含功能、精度（Loss值）、性能检查，检查通过打印"Run testcase success!"，失败则打印"Run testcase failed!"；

3、执行环境已预装软件包和Python3.7.5环境，调用命令"python3"、"python3.7"、"python3.7.5"均可，安装第三方库依赖使用"pip3"、"pip3.7"均可；

4、数据集和模型：小于500M的文件，建议使用`obsutil`命令下载(已预装)，过大的文件，建议提交Issue，注明数据集和下载地址，会提前下载到执行环境上,

已预置数据集&python第三方库: [Environments](https://gitee.com/ascend/modelzoo/blob/master/contrib/ENVIRONMENTS.md)

5、环境和其他问题，请提交Issue跟踪；

6、测试用例开发参考：
- [训练](https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/Transformer_for_TensorFlow)
- [离线推理](https://gitee.com/alexcheng88/modelzoo/tree/master/contrib/TensorFlow/Research/cv/efficientnet-b8/ATC_efficientnet-b8_tf_nkxiaolei)

 **五、PR提交**

- 关键要求：

1、请将modelzoo仓fork到个人分支，基于个人分支新增、修改和提交PR；

2、PR标题：线上活动，请在标题注明[线上贡献]；高校活动，请注明[xxx学校][高校贡献]；

3、built-in用户根据网络状态必须配置modelzoo_level.txt文件，且文件内容包含三个关键字段：FuncStatus(OK-流程通过/ **NOK-流程失败，不允许模型代码提交主仓** )；PerfStatus(OK-持平GPU/POK-0.5倍GPU/NOK-小于0.5倍GPU/PERFECT-1.2倍GPU)；PrecisionStatus(OK-精度达标,POK-Loss拟合但精度未实施， **NOK-Loss不拟合，不允许模型代码提交主仓** )；内容格式如下所示（注：“:”两侧无需空格，英文格式；）： 
            
```
            FuncStatus:OK/NOK
            PerfStatus:PERFECT/OK/POK/NOK
            PrecisionStatus:OK/POK/NOK
```
4、contrib用户根据网络状态必须配置modelzoo_level.txt文件，且文件内容包含关键字段：GPUStatus(OK-GPU复现/NOK-GPU未复现)； NPUMigrationStatus(OK-自动迁移成功/POK-自动迁移失败, 手写规避成功/NOK-均失败)； FuncStatus(OK-基础功能打通/NOK-基础功能失败，不允许模型代码提交到master)； PrecisionStatus(OK-精度达标/POK-Loss拟合但精度未完全达标/NOK-精度不达标, 不允许模型代码提交到master)； AutoTune(OK-性能持平或高于GPU/POK-性能有提升但低于GPU/NOK-性能无提升或者功能失败)； PerfStatus(训练：PERFECT-性能1.2倍GPU/OK-性能持平GPU/POK-性能0.5倍GPU/NOK-性能小于0.5倍GPU；推理：OK-4*310单卡>GPU/NOK-其它)； ModelConvert:OK/NOK(仅推理, OK-om转换成功/NOK-om转换失败)； QuantStatus:OK/NOK(仅推理, OK-精度损失1%以内，性能有提升/POK-性能有提升但未达标/NOK-量化失败)；

样例：modelzoo_level.txt文件 

-----仅限训练----- 


```
GPUStatus:OK/NOK
NPUMigrationStatus:OK/POK/NOK
```


-----仅限推理----- 

```
ModelConvert:OK/POK/NOK 
QuantStatus:OK/POK/NOK 
```

-----通用部分----- 

```
FuncStatus:OK/NOK 
PrecisionStatus:OK/POK/NOK 
AutoTune:OK/POK/NOK 
PerfStatus:PERFECT/OK/POK/NOK
```
5、网络名称命名规范：*_for_框架，注：*代表任意内容，如网络名称或网络名称+网络ID；

 **六、编程规范**

- 规范标准

1、C++代码遵循google编程规范：Google C++ Coding Guidelines；单元测测试遵循规范： Googletest Primer。

2、Python代码遵循PEP8规范：Python PEP 8 Coding Style；单元测试遵循规范： pytest

- 规范备注

1、优先使用string类型，避免使用char*；

2、禁止使用printf，一律使用cout；

3、内存管理尽量使用智能指针；

4、不准在函数里调用exit；

5、禁止使用IDE等工具自动生成代码；

6、控制第三方库依赖，如果引入第三方依赖，则需要提供第三方依赖安装和使用指导书；

7、一律使用英文注释，注释率30%--40%，鼓励自注释；

8、函数头必须有注释，说明函数作用，入参、出参；

9、统一错误码，通过错误码可以确认那个分支返回错误；

10、禁止出现打印一堆无影响的错误级别的日志；
