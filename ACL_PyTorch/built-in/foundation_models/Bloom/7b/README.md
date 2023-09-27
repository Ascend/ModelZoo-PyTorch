#   BLOOM-7B模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)


  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

BLOOM是 BigScience Large Open-science Open-access Mul-tilingual Language Model首字母的缩写。它是一个Transformer解码器（Decoder-Only）模型，在一个称之为ROOTS的语料库上训练出来的176B参数规模的自回归语言模型。BLOOM 也是第一个开源开放的超过100B的语言模型。

参考论文：

- [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/1810.04805)

参考实现：

- 模型结构

  ```
  https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bloom/modeling_bloom.py
  ```

- 模型配置和权重

  ```
  https://huggingface.co/bigscience/bloom-7b1/tree/main
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                | 数据排布格式 |
  | -------------- | -------- | ------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len | ND           |
  | attention_mask | INT64    | batchsize x seq_len | ND           |

- 输出数据

  | 输出数据     | 数据类型 | 大小                | 数据排布格式 |
  | ------------ | -------- | ------------------- | ------------ |
  | generate_ids | FLOAT32  | batchsize x seq_len | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                                | 版本         | 环境准备指导                                                                                                                                          |
  | ------------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                          | 23.0.T50     | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                                | 7.0.T3       | -                                                                                                                                                     |
  | PythonFrameworkAdaptor                                              | 5.0.RC3.B020 | -                                                                                                                                                     |
  | Torch                                                               | 1.11.0       | -                                                                                                                                                     |
  | 说明：Atlas 300I Duo/Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \            | \                                                                                                                                                     |

  **表 2** 硬件形态
   | CPU     | Device   |
   | ------- | -------- |
   | aarch64 | 300I DUO |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码。

   ```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master                                            # 切换到对应分支
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/foundation_models/bloom/7b           # 切换到模型的代码仓目录
   ```

2. 获取模型配置及权重文件，和第1步源码置于同级目录下。

   ```shell
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bigscience/bloom-7b1
   mv ./bloom-7b1 ./model
   代码克隆失败，可尝试把 `https` 替换成 `http` 重试或手动下载
   ```

3. 安装必要依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

4. 安装transformers加速库

   下载 ```ascend_acclib.zip```

   ```shell
   unzip ascend_acclib_bloom7b.zip
   ```

## 模型推理<a name="section741711594517"></a>

1. 配置环境变量

   ```shell
	source /usr/local/Ascend/ascend-toolkit/set_env.sh
	cd ascend_acclib_bloom7b
	source set_env.sh
   ```

2. 开始推理验证

   **300I DUO双芯**

   ```shell
   cut_model_and_run_bloom.sh
   ```
