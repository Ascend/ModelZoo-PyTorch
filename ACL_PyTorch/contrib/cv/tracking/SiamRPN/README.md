# SiamRPN模型-推理指导


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

SiamRPN是一个能实时完成视觉物体跟踪并呈现出顶级性能的基于深度学习的跟踪器网络。

- 参考实现：

  ```
  url=https://github.com/STVIR/pysot
  branch=master
  model_name=SiamRPN
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

  | 输入数据  | 数据类型   | 大小                      | 数据排布格式  |
  | -------- | --------- | ------------------------- | ------------ |
  | template | RGB_FP32  | batchsize x 3 x 127 x 127 | NCHW         |
  | search   | RGB_FP32  | batchsize x 3 x 255 x 255 | NCHW         |


- 输出数据

  | 输出数据  | 数据类型  | 大小                      | 数据排布格式  |
  | -------- | -------- | ------------------------- | ------------ |
  | cls      | FLOAT32  | batchsize x 10 x 25 x 25  | NCHW         |
  | loc      | FLOAT32  | batchsize x 20 x 25 x 25  | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套        | 版本    | 环境准备指导             |
| ---------- | ------- | ----------------------- |
| 固件与驱动  | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN       | 6.0.0   | -                       |
| Python     | 3.7.5   | -                       |
| PyTorch    | 1.8.0   | -                       |  

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

> **说明：** 
> 该推理指导的所有步骤都以具体路径和参数为例进行说明，若有修改需要，请对开源代码仓中的pysot/pysot/core/config.py配置文件以及推理过程中的所有路径和参数进行相应修改。

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   得到本项目代码后，将SiamRPN项目放置在/home目录下,进入/home/SiamRPN目录下，下载开源代码仓

   ```
   git clone https://github.com/STVIR/pysot.git
   ```

   确认获取的开源pysot项目文件存放在/home/SiamRPN目录下，进入/home/SiamRPN/pysot目录下执行

   ```
   patch -N -p1 < ../fix.patch
   ```

2. 安装依赖。

   ```
   cd /home/SiamRPN
   pip3 install -r requirements.txt
   cd pysot
   export PYTHONPATH=/home/SiamRPN/pysot:$PYTHONPATH
   python3 setup.py build_ext --inplace install
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    通过[Vot Challenge](https://votchallenge.net/)获取VOT2016数据集。

    将数据集VOT2016下载并放在/root/datasets目录下。
   
## 模型推理<a name="section741711594517"></a>

> **说明：** 
> 下述test/pth2onnx.sh、test/onnx2om.sh和test/eval_acc_perf.sh三个脚本中的环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      在[PySOT Model Zoo](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md)上获取权重文件siamrpn_r50_l234_dwxcorr。
       
   2. 导出onnx文件。

         ```
         cd /home/SiamRPN
         bash test/pth2onnx.sh
         ```

         获得SiamRPN.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      ```
      bash test/onnx2om.sh
      ```

      > **说明：** 
      > 请根据实际使用的芯片型号对脚本中的soc_version参数进行修改。

2. 开始推理验证。

    1. 安装ais_bench推理工具。

        请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

    2. 执行推理、精度验证与性能验证。

        ```
        bash test/eval_acc_perf.sh VOT2016 1
        ```
       参数说明：
       - VOT2016 使用的推理数据集
       - 1 使用的npu卡号

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

- 评测结果：

- 310精度和性能
```
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness |   Average   |  EAO  |
------------------------------------------------------------
|  VOT2016   |  0.639   |   0.177    |    42fps    | 0.483 |
------------------------------------------------------------
```

- 参考pth精度和性能
```
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness |   Average   |  EAO  |
------------------------------------------------------------
|  VOT2016   |  0.642   |   0.196    |    35fps    | 0.464 |
------------------------------------------------------------
```

  
- 性能计算方式： 
  fps计算方式为单位时间内处理的图片数量，即 图片数量 / 时间 。
  根据310单device需乘以4之后再和T4对比，故310单卡性能理论计算为42×4=168fps。

- 备注：
- (1) 310精度相较于T4下降0.3%，但鲁棒性和EAO均有提升。310单device的实际平均性能为42fps。T4单卡平均性能为35fps，由于运行场景等干扰因素不同，会导致结果有所浮动，35fps为多次测量后平均近似值，供参考。
- (2) 性能数据(speed)在推理过程中会展示，在推理结束后会展示平均性能(average speed)。
- (3) 本推理为视频追踪，输入对象为视频，故不设置多batch。 
 