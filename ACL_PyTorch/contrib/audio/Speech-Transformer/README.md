# Speech-Transformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&性能](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Speech-Transformer 是一种无递归的 encoder-decoder 网络结构，已经在小规模语音识别数据集上的识别工作中显现出了良好的结果。

- 参考实现：

  ```
  url=git clone https://github.com/kaituoxu/Speech-Transformer
  branch=master
  model_name=Speech-Transformer
  ``` 
 
  通过 Git 获取对应 commit_id 的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

encoder

- 输入数据

  | 输入数据       | 数据类型  | 大小           | 数据排布格式  |
  | ------------- | -------- | -------------- | ------------ |
  | padded_input  | FLOAT32  | 1 x 512 x 320  | ND           |
  | non_pad_mask  | FLOAT32  | 1 x 512 x 1    | ND           |
  | slf_attn_mask | BOOL     | 1 x 512 x 512  | ND           |

- 输出数据

  | 输出数据    | 数据类型  | 大小          | 数据排布格式  |
  | ---------- | -------- | ------------- | ------------ |
  | enc_output | FLOAT32  | 1 x 512 x 512 | ND           |

decoder

- 输入数据

  | 输入数据         | 数据类型  | 大小          | 数据排布格式  |
  | --------------- | -------- | ------------- | ------------ |
  | ys_in           | INT64    | 1 x 128       | ND           |
  | encoder_outputs | FLOAT32  | 1 x 512 x 512 | ND           |
  | non_pad_mask    | FLOAT32  | 1 x 128 x 1   | ND           |
  | slf_attn_mask   | UINT8    | 1 x 128 x 128 | ND           |

- 输出数据

  | 输出数据     | 数据类型  | 大小           | 数据排布格式  |
  | ----------- | -------- | -------------- | ------------ |
  | dec_output  | FLOAT32  | 1 x 128 x 512  | ND           |

tgt_word_prj

- 输入数据

  | 输入数据         | 数据类型  | 大小          | 数据排布格式  |
  | --------------- | -------- | ------------- | ------------ |
  | input           | FLOAT32  | 1 x 512       | ND           |

- 输出数据

  | 输出数据     | 数据类型  | 大小           | 数据排布格式  |
  | ----------- | -------- | -------------- | ------------ |
  | output      | FLOAT32  | 1 x 4233       | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套        | 版本    | 环境准备指导             |
| ---------- | ------- | ----------------------- |
| 固件与驱动  | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN       | 6.0.0 | -                       |
| Python     | 3.7.5   | -                       |
| PyTorch    | 1.8.0   | -                       |  

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/kaituoxu/Speech-Transformer
   cd Speech-Transformer
   patch -p1 < ../SpeechTransformer.patch
   cd ..
   ```

   **将所有文件放在开源仓 Speech-Transformer/egs/aishell 下。**

2. 安装依赖。

   1. 安装依赖包
      ```
      cd Speech-Transformer/egs/aishell
      pip3 install -r requirements.txt
      cd ../../../
      ```
   
   2. 安装kaldi
      ```
      git clone https://github.com/kaldi-asr/kaldi
      ```
      请根据kaldi/INSTALL文件中的Option 1进行安装。

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   1. 下载data_aishell

      在 [OpenSLR](http://www.openslr.org/33/) 下载数据集，解压 data_aishell.tgz，然后再解压 data_aishell/wav 目录下面所有的文件。

   2. 提取特征

      ```
      cd Speech-Transformer/tools

      # 指向kaldi的源码
      make KALDI=../../kaldi

      cd ../egs/aishell

      # 修改run.sh中data变量指向aishell数据集，例如，data_aishell放在/home目录下面，则修改data=/home
      bash run.sh
      ```
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用 PyTorch 将 .pth 模型权重文件转换为 .onnx 文件，再使用 ATC 工具将 .onnx 文件转为离线推理模型文件 .om 文件。

   1. 获取权重文件。

      ```
      wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/audio/speech-transformer/final.pth.tar
      ```
       
   2. 导出 onnx 文件。

         ```
         bash test/pth2onnx.sh
         ```

         获得 encoder.onnx、decoder.onnx 和 tgt_word_prj.onnx 文件。

   3. 使用 ATC 工具将 ONNX 模型转 OM 模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
             +--------------------------------------------------------------------------------------------+
             | npu-smi 22.0.0                       Version: 22.0.2                                       |
             +-------------------+-----------------+------------------------------------------------------+
             | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
             | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
             +===================+=================+======================================================+
             | 0       310P3     | OK              | 17.0         56                0    / 0              |
             | 0       0         | 0000:AF:00.0    | 0            934  / 23054                            |
             +===================+=================+======================================================+
         ```

      3. 执行 ATC 命令。

         ```
         bash test/onnx2om.sh
         ```

         运行成功后生成 encoder.om、decoder.om 和 tgt_word_prj.om 模型文件。

2. 开始推理验证。
   
   执行推理、精度验证和性能验证。

   ```
   bash test/eval_acc_perf.sh
   ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

评测结果：
| 模型 | 在线推理pth精度 | 310离线推理精度  |310P3离线推理精度  | 基准性能  | 310性能  | 310P3性能 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|  Speech-Transformer  |    9.8     | 9.9  | 10.8  |  0.83 fps | 0.82 fps | 1.71 fps |

备注：
1. 模型不支持多 batch
2. 精度测评脚本包含了精度和性能结果, 结果中 Err 即为精度
3. 基准性能获取方法

   ```
   pip3 install onnxruntime-gpu
   bash test/eval_acc_perf_onnx.sh
   ```