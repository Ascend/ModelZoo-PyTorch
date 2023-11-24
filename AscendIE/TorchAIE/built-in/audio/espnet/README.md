# Espnet_conformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section183221994400)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能精度](#ZH-CN_TOPIC_0000001172201573)






# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Espnet_conformer模型是一个使用conformer结构的ASR模型。

- 参考实现：

  ```
  url=git clone https://github.com/espnet/espnet
  branch=v.0.10.5
  model_name=tacotron2
  ```
  

通过Git获取对应commit\_id的代码方法如下：

```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

EspNet安装比较复杂，请参考https://espnet.github.io/espnet/installation.html

若安装mkl失败，则去launchpad.net/ubuntu/+source/intel-mkl/2020.0.166-1  

下载 intel-mkl_2020.0.166.orig.tar.gz 文件，解压后 bash install.sh安装即可

注意：mkl arm不适用于arm版本安装，推荐适用x86环境

## 输入输出数据<a name="section540883920406"></a>

- encoder输入数据

  | 输入数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input | input_dynamic_axes_1 x 83 | FLOAT32 | ND |


- encoder输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | 2863 | Add2863_dim_0x Add2863_dim_1 x 256 | FLOAT32  | ND |
  




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | -                                                       |
| Python                | 3.9.0           |                                                           
| torch                 | 2.0.1           |
| Ascend-cann-torch-aie | -               
| Ascend-cann-aie       | -
| 芯片类型                  | Ascend310P3     | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
安装依赖。

   om_gener安装

   ```
   git clone https://gitee.com/peng-ao/om_gener.git
   cd om_gener
   pip3 install .
   ```

   acl_infer安装

   ```
  git clone https://gitee.com/peng-ao/pyacl.git
  cd pyacl
  pip3 install .
   ```

## 获取源码<a name="section183221994400"></a>

在工作目录下执行下述命令获取源码并切换到相应路径。

请按照官方指导文档进行代码的安装

## 准备数据集<a name="section183221994411"></a>

在espnet/egs/aishell/asr1/文件夹下运行bash run.sh --stage -1 –stop_stage -1下载数据集

运行bash run.sh --stage 0 --stop_stage 0处理数据集

运行bash run.sh --stage 1 --stop_stage 1处理数据集

运行bash run.sh --stage 2 --stop_stage 2处理数据集

运行bash run.sh --stage 3 --stop_stage 3处理数据集

若缺少对应的文件夹，则自己建立文件夹

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   本模型基于开源框架PyTorch训练的Espnet_conformer进行模型转换。使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 在checkpoints目录下获取权重文件。

        下载路径：https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md
        
        对应Conformer(kernel size = 15) + SpecAugment + LM weight = 0.0下面的model link即可
        
        解压，将对应的conf，data, exp文件夹置于espnet/egs/aishell/asr1
        
   2. 导出torchscript模型，用于编译优化。

      1. 首先将export.py放在espnet根目录下，运行以下生成espnet_trace.ts
         ```
         python3 export.py --model_path egs/aishell/asr1/exp/train_sp_pytorch_train_pytorch_conformer_kernel15_specaug/results/model.last10.avg.best
         ```
      2. 运行以下命令编译模型 (注意：编译aie模型依赖的环境和espnet运行环境不同；编译环境参考“推理环境准备”)
         ```shell
         # 分档模型
         python3 compile.py --model_path=./espnet_trace.ts --flag=gear
         # 动态shape模型
         python3 compile.py --model_path=./espnet_trace.ts --flag=dynamic
         ```
         执行结束，会在当前目录下生成espnet_gear.ts, espnet_dynamic.ts, espnet_gear.om, espnet_dynamic.om文件。
         两个ts文件用于后续性能测试，两个om文件用于后续精度测试。

2. 开始推理验证。

   1. 获取精度

      ①静态shape

      首先修改acc.diff文件中的om模型路径（约162行）为生成的om路径

      ```shell
      cd espnet
      git checkout v.0.10.5
      patch -p1 < acc.diff
      cd espnet/egs/aishell/asr1
      bash acc.sh
      ```

      ②动态shape

      首先修改acc_dynamic.diff文件中的om模型路径（约162行）为生成的om路径

      ```shell
      cd espnet
      git checkout v.0.10.5
      patch -p1 -R < acc.diff  # 恢复上一个patch的影响
      patch -p1 < acc_dynamic.diff
      cd espnet/egs/aishell/asr1
      bash acc.sh
      ```

      即可打屏获取精度，精度保存在文件espnet/egs/aishell/asr1/exp/train_sp_pytorch_train_pytorch_conformer_kernel15_specaug/decode_test_decode_lm0.0_lm_4/result.txt

   2. 性能测试
      ```shell
      # 分档模型
      python3 perf_test.py --model_path=./espnet_gear.ts
      # 动态shape模型
      python3 perf_test.py --model_path=./espnet_dynamic.ts
      ```
      执行结束，会打印出性能结果。


# 模型推理性能精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用aclruntime推理计算，性能精度参考下列数据。

| 模型              | 310P性能             | 310P精度(Err) |
|-----------------|--------------------|-------------|
| Espnet_conformer | 分档：358fps；动态：55fps | 5.4%        |

