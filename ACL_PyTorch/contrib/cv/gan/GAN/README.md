# GAN模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)
  
  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)
  
  ---

# 概述

生成式对抗网络（GAN, Generative Adversarial Networks ）是一种深度学习模型，是近年来复杂分布上无监督学习最具前景的方法之一。模型通过框架中（至少）两个模块：生成模型（Generative Model，下文简写G）和判别模型（Discriminative Model，下文简写D）的互相博弈学习产生相当好的输出。判别模型（D）的任务就是判断一个实例是真实的还是由模型生成的；生成模型（G）的任务就是生成一个实例来骗过判别模型（D）。两个模型相互对抗，最后会达到一个平衡，即生成模型生成的实例与真实的没有区别，判别模型无法区分输入数据是生成的还是原始真实的数据。

- 参考实现：
  
  ```
  url=https://github.com/eriklindernoren/PyTorch-GAN
  commit_id=36d3c77e5ff20ebe0aeefd322326a134a279b93e
  ```
  
  通过Git获取对应commit_id的代码方法如下：
  
  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 输入输出数据

- 输入数据
  
  | 输入数据  | 数据类型 | 大小              | 数据排布格式 |
  | ----- | ---- | --------------- | ------ |
  | input | FP32 | batchsize x 100 | NCHW   |

- 输出数据
  
  | 输出数据   | 大小                | 数据类型 | 数据排布格式 |
  | ------ | ----------------- | ---- | ------ |
  | output | batchsizex1x28x28 | FP32 | NCHW   |

# 推理环境准备[所有版本]

- 该模型需要以下插件与驱动
  
  **表 1** 版本配套表

| 配套                                        | 版本      | 环境准备指导                                                                                        |
| ----------------------------------------- | ------- | --------------------------------------------------------------------------------------------- |
| 固件与驱动                                     | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                      | 5.1.RC2 | -                                                                                             |
| Python                                    | 3.7.5   | -                                                                                             |
| PyTorch                                   | 1.5.0   | -                                                                                             |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                             |

# 快速上手

1. 获取源码
   
   ```
   git clone https://github.com/eriklindernoren/PyTorch-GAN
   ```

2. 安装依赖
   
   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）
   
   由于源代码中未提供测试数据,这里调用GAN_preprocess.py来生成测试数据，保存到vectors文件夹下。

2. 数据预处理。
   
   将原始数据集转换为模型输入的二进制数据。执行GAN_preprocess.py脚本。
   
   ```
   python3.7 GAN_preprocess.py --online_path=images --offline_path=vectors --pth_path=generator_8p_0.0008_128.pth --iters 100 --batch_size 64
   ```
   
   > **说明：** 
   > 该命令每个bs都要执行一次,每次执行时需修改batchsize参数。
   
   参数说明
   
   - --online_path：生成的数据集路径
   
   - --pth_path：权重文件路径
   
   - --offline_path：bin文件路径
   
   - --batch_size:输入批次
   
   - --iters:输入参数
     
     生成“batch_size”为64的模型输入，得到的bin文件储存在当前目录下“vectors”文件夹中

## 模型推理

1. 模型转换。
   
   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
   
   1. 获取权重文件。
      
      [GAN预训练pth权重文件](https://wws.lanzoui.com/ikXFJvljkab)
      
      下载解压至当前工作目录
   
   2. 导出onnx文件。
      
      1. 使用GAN_pth2onnx.py导出onnx文件。
      
      运行GAN_pth2onnx.py”脚本。
      
      ```
      python3.7 GAN_pth2onnx.py --input_file=generator_8p_0.0008_128.pth --output_file=GAN.onnx
      ```
      
      获得“GAN.onnx”文件。
      
      2. 优化ONNX模型。
      
      ```
      python3 -m onnxsim GAN.onnx GAN_sim.onnx
      ```
      
      获得GAN_sim.onnx文件。
   
   3. 使用ATC工具将ONNX模型转OM模型。
      
      1. 配置环境变量。
      
      ```
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```
      
      > **说明：** 
      > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
      2. 执行命令查看芯片名称（${chip_name}）。
      
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
      
      3. 执行ATC命令。
      
      ```
      atc --framework=5 --model=GAN_sim.onnx --output=GAN_bs1  --input_format=NCHW --input_shape="Z:1,100" --log=error --soc_version=Ascend${chip_name} 
      ```
      
      - 参数说明：
      
      - --model：为ONNX模型文件。
      
      - --framework：5代表ONNX模型。
      
      - --output：输出的OM模型。
      
      - --input_format：输入数据的格式。
      
      - --input_shape：输入数据的shape。
      
      - --log：日志级别。
      
      - --soc_version：处理器型号。
      
      运行成功后生成“GAN_bs1.om”模型文件。

2. 开始推理验证。
   
   a. 安装ais_bench推理工具。
   
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  
   
   b. 执行推理。
   
   创建推理输出的文件夹
   
   ```
   mkdir result
   ```
   
   ```
   python3.7 -m ais_bench --model "GAN_bs1.om" --input ./vectors --output "./result"  --batchsize 1 --loop 100
   ```
   
   - 参数说明：
     
     - --model：om文件路径。
     - --input:输入路径。
     - --output：输出路径。
     
     推理后的输出默认在当前目录result下。

     
     c. 精度验证。
     
     调用GAN_postprocess.py来进行后处理,详细的结果输出在genimg文件夹中，可以和images文件夹下的在线推理结果做对比，看谁生成的图片质量更好。
     
     ```
     python3.7 GAN_postprocess.py --txt_path=./result/2022_11_14-14_02_51 --infer_results_path=genimg
     ```
     
     **说明：** 
     因在线推理不支持bs1，所以该模型不支持bs1的精度验证。
     
     d. 性能验证。
     
     ais_infer纯推理验证不同batch_size的om模型的性能，参考命令如下:
     
     ```
     python3.7 -m ais_bench --model=${om_model_path} --loop=100 --batchsize=${batch_size}
     ```

# 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

|         | 310        | 310P3     | 310P3/310   |
| ------- | ---------- | --------- | ----------- |
| bs1     | 12,121.21  | 9007.47  | 0.949366441 |
| bs4     | 48,484.84  | 39,253    | 0.912711685 |
| bs8     | 94,117.64  | 79,647    | 0.899375717 |
| bs16    | 206,451.61 | 160,191   | 0.867958308 |
| bs32    | 387,878.78 | 296,927   | 0.799031465 |
| bs64    | 711,111.11 | 469,239   | 0.697836685 |
|         |            |           |             |
| 最优batch | 711111.11  | 496239.42 | 0.697836685 |
上述表格中中310P3芯片性能数据为loop值为10下取得的性能数据，较高的loop值测试下，性能数据往往高于表格数据
源码中未有精度对比部分，这里以两种不同的方式对同一输入的输出结果对比为准。将离线推理结果和在线推理结果做对比，看得出离线推理生成的图片质量更好。
