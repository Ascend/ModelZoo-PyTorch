# Baichuan2-7B-Chat模型-推理指导

- [概述](#概述)

- [输入输出数据](#输入输出数据)

- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [获取源码及依赖](#获取源码及依赖)
  - [模型推理](#模型推理)

- [模型推理性能](#模型推理性能)

# 概述

   Baichuan2是一个大语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 参考实现：
   ```bash
   # Baichuan2
   https://github.com/baichuan-inc/Baichuan2
   ```

# 输入输出数据
- 输入数据

  | 输入数据      | 大小          | 数据类型  | 数据排布格式 | 是否必选 |
  |-----------|-------------|-------|--------|------|
  | input_ids | 1 x SEQ_LEN | INT64 | ND     | 是    |
  | atten_mask | 1 x 1 x SEQ_LEN x SEQ_LEN | float32 | ND     | 否|

- 输出数据

  | 输出数据       | 大小                 | 数据类型  | 数据排布格式 |
  |------------|--------------------|-------|--------|
  | output_ids | 1 x OUTPUT_SEQ_LEN | INT64 | ND     |

- 约束   
  `SEQ_LEN + OUTPUT_SEQ_LEN <= MAX_SEQ_LEN`，`MAX_SEQ_LEN`由启动时配置

# 推理环境准备

 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套             | 版本       | 下载链接                                                                                                                                                                              |
  |----------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | 固件与驱动          | 23.0.T50 | [HDK固件与驱动](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-252764743/software/261159044?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743)     |
  | CANN           | 7.0.T3   | [CANN toolkit](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261075821?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
  | Python         | 3.7.5    | -                                                                                                                                                                                 |         
  | PytorchAdapter | 1.11.0   | [PTA安装](https://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/ascend/pytorch/version_compile/202308/20230818_05/ubuntu_x86/torch_v1.11.0.tar.gz)            |
  | 推理引擎           | 7.0.T6   | [推理引擎](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261177692?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892969%7C251168373)         |

  **表 2** 推理引擎依赖

   | 软件     | 版本要求      | 
   |--------|-----------|
   | glibc  | \>= 2.27  | 
   | gcc    | \>= 7.5.0 | 

  **表 3** 硬件形态

   | CPU    | Device   |
   |--------|----------|
   | x86_64 | 300I DUO |

# 快速上手

## 获取源码及依赖

1. 安装环境

- 安装CANN环境
   1. 安装HDK   
   下载[固件与驱动](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-252764743/software/261159044?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743)，文件列表如下：
   ```bash
   Ascend-hdk-310p-npu-driver_{version}_linux-{arch}.run
   Ascend-hdk-310p-npu-firmware_{version}.run
   ```
   安装命令：
   ```bash
   chmod +x Ascend-hdk-310p-npu-firmware_{version}.run
   chmod +x Ascend-hdk-310p-npu-driver_{version}_linux-{arch}.run
   ./Ascend-hdk-310p-npu-firmware_{version}.run --install
   ./Ascend-hdk-310p-npu-driver_{version}_linux-{arch}.run --install
   ```
     
   2. 安装CANN   
   下载[CANN toolkit及kernel](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261075821?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)，文件列表如下：
   ```bash
   Ascend-cann-toolkit_{version}_linux-{arch}.run
   Ascend-cann-kernels-310p_{version}_linux.run
   ```
   安装命令同HDK   
   
   3. 安装PytorchAdapter
   下载[PytorchAdapter](https://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/ascend/pytorch/version_compile/202308/20230818_05/ubuntu_x86/torch_v1.11.0.tar.gz)，文件列表：
   ```bash
   pytorchv1.11.0.tar.gz
   ```
   安装命令：
   ```bash
   tar -zxvf pytorchv1.11.0.tar.gz #解压缩
   pip3 install torch-{version}-linux_{arch}.whl
   pip3 install torch_npu-{version}-linux_{arch}.whl
   pip3 install apex-0.1_ascend_{version}-linux_{arch}.whl
   ```
  
   4. 安装依赖   
   参考[推理环境准备](#推理环境准备)安装配套软件。安装python依赖。
  ```bash
   pip3 install -r requirements.txt
   ```

2. 下载模型权重，放置到自定义`MODEL_PATH`
   ```bash
   # Baichuan2 7B模型
   https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
   # 放置模型到MODEL_PATH
   ```

3. 安装加速库
   下载[推理引擎文件](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261177692?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892969%7C251168373)，`ascend_acclib.zip`
   ```bash
   # 解压
   unzip ascend_acclib.zip
   cd ascend_acclib
   source set_env.sh
   ```

4. 代码修改

- 拷贝`modeling_baichuan_ascend.py`到`MODEL_PATH`
- 修改`config.json`为`modeling_baichuan_ascend.py`
  ```bash
  "auto_map": {
    "AutoConfig": "configuration_baichuan.BaichuanConfig",
    "AutoModelForCausalLM": "modeling_baichuan_ascend.BaichuanForCausalLM"
  }
  ```
- 修改`run_baichuan2_7b.py`中`MODEL_PATH`为真实`MODEL_PATH`

## 模型推理

- 执行模型运行
   ```bash
   python3 run_baichuan2_7b.py
   ```
   该命令会运行两次离线推理实例，并启动chat对话，默认开启`stream`对话，输入`stream`切换流逝对话状态，输入`exit`退出对话并退出程序。
- 自定义运行可参考`run_baichuan2_7b.py`

# 模型推理性能

| 硬件形态  |   输入长度   |  输出长度  |     解码速度      |
|:-----:|:--------:|:------:|:-------------:|
| Duo单芯 | 1 x 2048 |   64   | 6.34 tokens/s |
| Duo双芯 | 1 x 2048 | 64 | 开发中 |



