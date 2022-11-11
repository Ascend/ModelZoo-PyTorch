# DQN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


  ******   


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DQN其实是深度学习和强化学习知识的结合，也就是用Deep Networks框架来近似逼近强化学习中的Q value。DQN相比Q-learning，DQN可以在每一步进行多个权重的更新。由于样本之间的强相关性，直接从连续样本中学习效率低效；随机化样本会打破这些相关性，从而减少更新的方差。当学习策略时，当前参数确定参数训练的下一个数据样本。

- 参考实现：

  ```
  url=https://github.com/ShangtongZhang/DeepRL
  branch=master
  commit_id=13dd18042414ad112bd0bd383a836d8d739e8acf
  model_name=DQN
  ```
  适配昇腾 AI 处理器的实现：
   
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch
  tag=v.0.4.0
  code_path=ACL_PyTorch/contrib/rl
  ```

  通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | state    | FLOAT32  | batchsize x 1 x4x84x84    | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | action   |2 or 3    | FLOAT32  | N          |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
    git clone https://github.com/ShangtongZhang/DeepRL
    mv DeepRL/deep_rl deep_rl
   ```    


2. 安装依赖。

   ```
    pip3 install -r requirements.txt    
    git clone https://github.com/openai/baselines.git
    cd baselines
    pip install -e .
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型没有原始输入的数据集，故而将在线推理的state输出保存作为数据集。

2. 数据预处理。

   将在线推理生成的state输出保存为pt文件。
   
   运行“dqn_preprocess.py”、“get_dataset_bin.py”获取state文件并将其转换为bin文件。

   ```
   mkdir state_path
   mkdir bin_path
   python3 dqn_preprocess.py --pth-path='dqn.pth' --state-path='state_path' --num=20
   python3 get_dataset_bin.py --state-path='state_path' --bin-path='bin_path'
   ```
   -   参数说明：

       -   --pth-path：输入的pth权重文件。
       -   --state-path：输出的state模型文件。
       -   --num：保存state文件的数量。
       -   --bin-path：转换得到的bin文件。


   > **说明：** 
     > 源码在文件DeepRL\deep_rl\component\relay.py中205行等多处使用到名为async的变量名与python3的关键字冲突，需要把这些async替换为一个不冲突的变量名。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件：“dqn.pth”。

   2. 导出onnx文件。

      使用“dqn.pth”导出onnx文件。

      运行dqn_pth2onnx.py脚本。

         ```
         python3 dqn_pth2onnx.py --pth-path="dqn.pth" --onnx-path="dqn.onnx"
         ```
        

       获得dqn.onnx文件。
       
         

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
           source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3（自行替换）
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
          atc --framework=5 --model=dqn.onnx --output=dqn_bs1 --input_format=NCHW --input_shape="input:1,4,84,84" --log=error --soc_version=${chip_name}
         ```

         - 参数说明：

           -   --framework：5代表ONNX模型。           
           -   --model：为ONNX模型文件。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

         运行成功后生成dqn_bs1.om模型文件。



2. 开始推理验证。

   1. 使用ais-infer工具进行推理。
   
    ais-infer工具获取及使用方式请点击查看[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

    
   2. 执行推理。

       ```
        mkdir result
        python3.7 ais_infer.py --model=dqn_bs1.om --input bin_path --output result --outfmt BIN --batchsize 1
       ```
       
        -   参数说明
            -   --model：输入的om文件。
            -   --input：输入的bin数据文件。
            -   --output: 推理结果输出路径。
            -   --outfmt：输出数据的格式。
            -   --batchsize：训练批次大小。
 
        > **说明：** 
        > 执行ais-infer工具请选择与运行环境架构相同的命令。
 

   3. 精度验证。

      选取20个step，对比在线推理与离线推理的action的结果，以此作为模型精度。
      
      进行后处理并将处理得到的数据进行精度对比

      ```
       python3.7 dqn_postprocess.py --pth-path='dqn.pth' --state-path='state_path' --outbin-path='result/2022_11_8_23_23_50' --num=20
      ```

      -   参数说明：

          -   --pth-path：输入的pth权重文件。
          -   --state-path：输出的state模型文件。
          -   --outbin-path：推理输出结果的路径。
          -   --num：参数输出比较的个数。

    4.性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
       python3.7 ais_infer.py --model=dqn_bs1.om --loop=20 --batchsize 1
      ```
      
      -   参数说明
          -   --model：输入的om文件。
          -   --batchsize：训练批次大小。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能和精度参考下列数据。

| 芯片型号 | Batch Size   | 310P性能 | 310P精度 |
| --------- | ---------------- | -------------- | -------------- |
|Ascend310P3| BS1              | 8147.444   |100%|
 


注：此模型不支持多batch。
