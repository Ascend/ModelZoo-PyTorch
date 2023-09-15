# LLaMA-Adaptor模型-推理指导

- [概述](#概述)

- [输入输出数据](#输入输出数据)

- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [获取源码及依赖](#获取源码及依赖)                                                                                                          
  - [模型推理](#模型推理)

- [模型推理性能](#模型推理性能)

# 概述

   LLaMA（Large Language Model Meta AI），由 Meta AI 发布的一个开放且高效的大型基础语，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。LLaMA-Adapter作为一个通用的多模态基础模型，它集成了图像、音频、文本、视频和3D点云等各种输入，同时还能提供图像、文本和检测的输出。本文旨在指导在Ascend产品上实现LLaMA-Adapter的完整推理流程。

- 参考实现：
   ```bash
   https://github.com/OpenGVLab/LLaMA-Adapter
   ```

# 输入输出数据
- 输入数据

  | 输入数据      | 大小          | 数据类型  | 数据排布格式 | 是否必选 |
  |-----------|-------------|-------|--------|------|
  | image | BATCH_SIZE x 224 x 224 x 3 | FLOAT16 | NHWC     | 是    |
  | prompt | BATCH_SIZE x SEQ_LEN | INT64 | ND     | 否|

- 输出数据

  | 输出数据       | 大小                 | 数据类型  | 数据排布格式 |
  |------------|--------------------|-------|--------|
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64 | ND     |


# 推理环境准备

 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套             | 版本       | 下载链接                                                                                                                                                                                 |
|----------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 固件与驱动          | 23.0.T50 | [https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/261159044?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fascend-computing%2Fascend-hdk-pid-252764743%2Fsoftware%2F261159044%3FidAbsPath%3Dfixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
| CANN           | 7.0.T3   | [https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261075821?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fascend-computing%2Fcann-pid-251168373%2Fsoftware%2F261075821%3FidAbsPath%3Dfixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
|                |          |                                                              |
| PytorchAdapter | 1.11.0   | -[https://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/ascend/pytorch/version_compile/202308/20230818_05/centos_aarch64/torch_v1.11.0.tar.gz](https://gitee.com/link?target=https%3A%2F%2Fcontainer-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com%2Fpackage%2Fascend%2Fpytorch%2Fversion_compile%2F202308%2F20230818_05%2Fcentos_aarch64%2Ftorch_v1.11.0.tar.gz) |
| 加速库           | -   |     https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261249334?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373     |
| 基础镜像 | - | https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/docker_file/chatglm6b/acltransformer_base.tar |
| Python | 3.7.5 | - |

  **表 3** 硬件形态

   | CPU    | Device   |
   |--------|----------|
   | aarch64 | Ascend310P3 |

# 快速上手

## 获取源码及依赖

1. 环境部署
- 安装HDK

- 安装CANN

- 基础镜像搭建 

   [获取镜像](https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/docker_file/chatglm6b/acltransformer_base.tar)

   ```bash
   docker load < acltransformer_base.tar
   ```

   基于以上镜像启动docker，注意，请确保docker中的网络代理、git代理等配置正常，以方便调试。

- 安装PytorchAdapter
   docker中自带conda环境py3.7，需要安装与环境中cann包以及hdk匹配的pta中的torch、torch_npu

   ```
   pip install {name}.whl
   ```

- 安装依赖  
   参考[推理环境准备](#推理环境准备)安装配套软件。安装python依赖。

   ```bash
   pip3 install -r requirements.txt
   ```

2. 下载LLaMA模型权重，放置到自定义`input_dir`
   | 权重    | 下载地址                                                     |
   | ------- | ------------------------------------------------------------ |
   | llama1  | https://ai.meta.com/resources/models-and-libraries/llama-downloads/ |
   | BIAS-7B | https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth |
   
3. 安装加速库
   下载
   
   [加速库](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261249334?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)
   
   `ascend_acclib_llama_adapter.zip`
   
   ```bash
   # 解压
   unzip ascend_acclib_llama_adapter.zip
   cd ascend_acclib_llama_adapter
   source set_env.sh
   ```

## 模型推理

**模型的推理及执行都需在docker中进行**

1. 图像模型CLIP部分
 
    权重涉及半精度float16，暂时只支持在gpu环境导出onnx。
- 转onnx
   ```
   pth2onnx.py ${pth_path}
   ```
   参数是上文指定下载好的BIAS-7B权重文件路径。
   执行该脚本后得到clip.onnx.

- 转om
   1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

   2. 执行命令查看芯片名称（$\{chip\_name\}）。

      ```
      npu-smi info
      #该设备芯片名为Ascend310P3 （自行替换）
      回显如下：
      +-------------------+-----------------+------------------------------------------------------+
      | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
      | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
      +===================+=================+======================================================+
      | 0       310P3     | OK              | 15.8         42                0    / 0              |
      | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
      +===================+=================+======================================================+
      | 1       310P3     | OK              | 15.4         43                0    / 0              |
      | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
      +===================+=================+======================================================+
      ```
      ```bash
      atc --model clip.onnx \
         --framework 5 \
         --output clip \
         --input_shape "input:${batch_size},224,224,3" \
         --input_format ND \
         --device 0 \
         --soc_version Ascend${chip_name} \
         --log info \
         --optypelist_for_implmode="Sigmoid,Gelu" \
         --op_select_implmode=high_performance
      ```
   - 参数说明：

      -   --model：为ONNX模型文件。
      -   --framework：5代表ONNX模型。
      -   --output：输出的OM模型。
      -   --input\_format：输入数据的格式。
      -   --optypelist_for_implmode：设置optype列表中算子的实现模式。
      -   --op_select_implmode：设置网络模型中算子的实现模式。
      -   --input\_shape：输入数据的shape。
      -   --log：日志级别。
      -   --soc\_version：处理器型号。
   
   执行完atc命令后得到clip.om模型文件
   
   需要安装ais_bench推理工具。
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

2. llama部分

   - 项目

     本项目具体目录为该工程下的llama_adapter_v2_multimodal7b

     后续替换文件关注该文件夹下的demo.p以及./llama/llama_adapter.py

     ```
     LLaMA-Adapter
       |-llama_adapter_v2_multimodal7b
          |-demo.py
          |-llama
             |-llama_adapter.py  
     ```

   - 脚本替换
     ModelZoo https://gitee.com/ascend/ModelZoo-PyTorch 

     目录：ModelZoo-PyTorch/ACL_PyTorch/built-in/foundation_models/llama_adapter

     1、llama_adapter_bsz.py拷贝到LLaMA-Adapter/llama_adapter_v2_multimodal7b/llama/

     2、demo_bsz.py拷贝到LLaMA-Adapter/llama_adapter_v2_multimodal7b/

   - 运行

     在LLaMA-Adapter/llama_adapter_v2_multimodal7b/目录下操作：

     先备份源码：

     ```shell
     cp ./llama/llama_adapter.py ./llama/llama_adapter_bak.py
     ```

     执行修改后脚本

     ```
     cp ./llama/llama_adapter_bsz.py ./llama/llama_adapter.py
     python demo_bsz.py
     ```

     demo_bsz.py说明：

     | 变量          | 说明                                                       |
     | ------------- | ---------------------------------------------------------- |
     | DEVICE_ID     | npu 卡号                                                   |
     | IMG_SIZE      | 图片前处理shape（224）                                     |
     | BATCH_SIZE    | batch_size 可调整为1或5                                    |
     | INFER_LOOP    | 循环次数，根据图片数目设置，样例单batch时为25，5batch时为5 |
     | LLAMA_DIR     | llama权重文件位置                                          |
     | CLIP_DIR      | clip的om模型位置                                           |
     | BIAS_DIR      | BIAS-7B.pth文件位置                                        |
     | PIC_FILE_PATH | 测试图片文件夹位置，样例的图片命名为: pic_{序号}.jpg       |

   - 性能打点

   - llama_adapter_bsz.py中添加了以下时间戳：

     ```
     self.clip_time		clip的om推理时间
     self.pre_time		llama的前处理时间
     self.forward_time	llma推理时间
     self.post_time		llama后处理时间
     self.decode_time	输出文本解码时间
     ```

     可于demo_bsz.py中每次调用model.generate()时取用，以打印显示其耗时情况


# 模型推理性能

| 硬件形态  |   batch_size   | 处理速率 |
|:-----:|:--------:|:------:|
| Ascend310P3 | 1 | 6.06 pictures/s |
| Ascend310P3 | 5 | 12.47 pictures/s |

 

