# SiamMask模型-推理指导


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

SiamMask是一种实时进行视觉物体跟踪和半监督视频物体分割的方法。

- 参考实现：

  ```
  url=https://github.com/foolwood/SiamMask
  branch=master
  model_name=SiamMask
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

mask

- 输入数据

  | 输入数据  | 数据类型   | 大小                      | 数据排布格式  |
  | -------- | --------- | ------------------------- | ------------ |
  | template | RGB_FP32  | batchsize x 3 x 127 x 127 | NCHW         |
  | search   | RGB_FP32  | batchsize x 3 x 255 x 255 | NCHW         |

- 输出数据

  | 输出数据      | 数据类型  | 大小                       | 数据排布格式  |
  | ------------ | -------- | -------------------------- | ------------ |
  | score        | FLOAT32  | batchsize x 10 x 25 x 25   | NCHW         |
  | delta        | FLOAT32  | batchsize x 20 x 25 x 25   | NCHW         |
  | mask         | FLOAT32  | batchsize x 3969 x 25 x 25 | NCHW         |
  | f0           | FLOAT32  | batchsize x 64 x 125 x 125 | NCHW         |
  | f1           | FLOAT32  | batchsize x 256 x 63 x 63  | NCHW         |
  | f2           | FLOAT32  | batchsize x 512 x 31 x 31  | NCHW         |
  | corr_feature | FLOAT32  | batchsize x 256 x 25 x 25  | NCHW         |

refine

- 输入数据

  | 输入数据 | 数据类型  | 大小                      | 数据排布格式  |
  | ------- | -------- | ------------------------- | ------------ |
  | p0      | FLOAT32  | batchsize x 64 x 61 x 61  | NCHW         |
  | p1      | FLOAT32  | batchsize x 256 x 31 x 31 | NCHW         |
  | p2      | FLOAT32  | batchsize x 512 x 15 x 15 | NCHW         |
  | p3      | FLOAT32  | batchsize x 256 x 1 x 1   | NCHW         |

- 输出数据

  | 输出数据      | 数据类型  | 大小              | 数据排布格式  |
  | ------------ | -------- | ----------------- | ------------ |
  | mask         | FLOAT32  | batchsize x 16129 | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套        | 版本    | 环境准备指导              |
| ---------- | ------- | ------------------------ |
| 固件与驱动  | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN       | 6.0.0   | -                        |
| Python     | 3.7.5   | -                        |
| PyTorch    | 1.8.0   | -                        |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/foolwood/SiamMask.git -b master 
   cd SiamMask
   git reset 0eaac33050fdcda81c9a25aa307fffa74c182e36 --hard

   cd utils/pyvotkit
   python3 setup.py build_ext --inplace
   cd ../../

   cd utils/pysot/utils/
   python3 setup.py build_ext --inplace
   cd ../../../../
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   获取VOT2016数据集，参见[Testing->Download test data](https://github.com/foolwood/SiamMask)

   请参考脚本中的命令构造数据集结构，如果数据集已存放到固定路径，通过ln -s软链接到`./SiamMask/data/VOT2016`，并将数据集中的VOT2016.json文件软链接到`./SiamMask/data/VOT2016.json`。
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载权重文件SiamMask_VOT.pth，放到当前目录。

      ```
      wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
      ```
      
   2. 导出onnx文件，使用ATC工具将ONNX模型转OM模型。

      ```
      bash test/pth2om.sh
      ```
      
      > **说明**：请根据CANN软件包实际安装路径以及实际使用芯片名称对脚本进行对应修改。

2. 开始推理验证。

    1. 安装ais_bench推理工具。

        请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

    2.  执行推理。

        ```
        bash test/eval_acc_perf.sh
        ```
        
        > **说明**：请根据CANN软件包实际安装路径对脚本进行对应修改。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能和精度参考下列数据。

| 模型     |                  官网pth精度                   | 310P3离线推理精度  | 基准性能    |  310P3性能  |
| :------: | :-------------------------------------------: | :---------------: | :--------: | :---------: |
| SiamMask | [0.433](https://github.com/foolwood/SiamMask) |      0.427        | 91.41 FPS  | 1673.94 FPS |

### 备注

- 由于pytorch与onnx框架实现引起的bn算子属性epsilon，momentum在两种框架下略微不同，onnx精度与pth精度差了一点，但是om与onnx精度一致。
- 因为SiamMask是前后帧连续处理，即上一帧的输入作为下一帧的输出，在Refine模块中进行动态pad，因此模型需要拆分为两段。因为SiamMask部分卷积存在使用自定义kernel来对输入进行卷积操作，导致卷积存在kernel和input的双输入的情况，[issue](http://github.com/onnx/onnx-tensorrt/issues/645)，故在线推理测试性能。计算的是拆分的两部分模型合在一起完整推理的性能。
- 因为SiamMask在corr部分固定了reshape之后的形状，并且针对前后帧连续处理，所以模型不支持多batch。