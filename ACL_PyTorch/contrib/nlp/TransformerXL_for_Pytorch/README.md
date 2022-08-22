# Transformer-XL模型-推理指导

- [Transformer-XL模型-推理指导](#transformer-xl模型-推理指导)
- [概述<a name="ZH-CN_TOPIC_0000001172161501"></a>](#概述)
  - [输入输出数据<a name="section540883920406"></a>](#输入输出数据)
- [推理环境<a name="ZH-CN_TOPIC_0000001126281702"></a>](#推理环境)
- [快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>](#快速上手)
  - [获取源码<a name="section183221994411"></a>](#获取源码)
  - [环境搭建<a name="section183221994411"></a>](#环境搭建)
  - [准备数据集<a name="section183221994411"></a>](#准备数据集)
  - [模型推理<a name="section741711594517"></a>](#模型推理)
- [推理结果<a name="ZH-CN_TOPIC_0000001172201573"></a>](#推理结果)
# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>
Transformer-XL是一个自然语言处理框架，在Transformer的基础上提出片段级递归机制(segment-level recurrence mechanism)，引入一个记忆(memory)模块（类似于cache或cell），循环用来建模片段之间的联系。并引入了相对位置编码机制(relative position embedding scheme)，代替绝对位置编码，能够更好地捕获长期依赖关系。
- 参考实现：

  ```
  url=https://github.com/kimiyoung/transformer-xl
  commit_id=44781ed21dbaec88b280f74d9ae2877f52b492a5
  model_name=Transformer-XL
  ```
  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | INT64   | 80 x 1       | ND         |
  | mems     | FLOAT16 | 160 x 1 x 512| ND         |


- 输出数据

  | 输出数据  | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | 80 x 204      | FLOAT16  | ND           |
  | mems    | 160 x 1 x 512 | FLOAT16  | ND           |


# 推理环境<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

   **表 1**  版本配套表

   | 配套 | 版本 | 环境准备指导 |
   | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
   | CANN | 5.1.RC1 | - |
   | python | 3.8以上 | - |
   | torch | 1.7.0   | - |
   | torchvision | 0.8.0 | - |
   | numpy | 1.22.0以上 | - |
   | ONNX | 1.8.0  | - |
   | tqdm | 4.60.0  | - |
   | onnx-simplifier | 0.3.10  | - |
   | protobuf | 3.19.0  | - |
   | decorator | 5.1.1  | - |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
## 获取源码<a name="section183221994411"></a>
1. 获取源码

   a. 从github上获取开源仓代码

      ```
      git clone https://github.com/kimiyoung/transformer-xl.git
      cd transformer-xl                                   
      git checkout 44781ed21dbaec88b280f74d9ae2877f52b492a5
      ```
   
   b. 从gitee上获得源码，覆盖掉开源仓代码
      ```
      git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
      cp -r ModelZoo-PyTorch/ACL_PyTorch/contrib/nlp/TransformerXL_for_Pytorch/. pytorch/
      patch -p1 < pytorch/sample.patch
      cd pytorch
      ```

## 环境搭建<a name="section183221994411"></a>
1. 创建虚拟环境并安装相关依赖包
   ```
   conda create -n 名字 python=3.8
   pip3 install -r requirements.txt
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集

   本模型支持enwik8数据集，在工作目录执行下述命令
   ```
   mkdir -p data
   cd data
   mkdir -p enwik8
   cd enwik8
   ```
   请用户需自行获取 [enwik8](http://cs.fit.edu/~mmahoney/compression/enwik8.zip) 数据集压缩包上传到enwik8文件夹路径下
   ```
   ├── pytorch
   │    ├── data
   │        ├── enwik8
   │            ├── enwik8.zip
   │    ├── ...
   ```
2. 数据预处理

   执行prep_enwik8.py脚本，完成预处理
   ```
   cd ../../
   python3 prep_enwik8.py
   ```
   从test数据集中抽取一部分进行评测
   ```
   cd data/enwik8
   rm -f cache.pt
   mv test.txt test.txt.bak
   head -1000 test.txt.bak > test.txt
   cd ../../
   ```
   运行成功后，生成如下文件目录
   ```
   ├── pytorch
   │    ├── data
   │        ├── enwik8
   │            ├── enwik8.zip
   │            ├── ...
   │            ├── test.txt
   │    ├── ...
   ```


## 模型推理<a name="section741711594517"></a>
1. 模型转换

   本模型基于开源TransformerXL模型训练后的权重进行模型转换。将模型权重文件.pt转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件
   
      浏览[昇腾ModelZoo](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/09454ba00de64604876888ddd8bc7d13) 网站，点击立即下载，并解压得到文件夹workdir0-enwik8（内含模型权重文件model.pt），并将文件夹workdir0-enwik8上传至工作路径下。
      ```
      ├── pytorch
      │    ├── data
      │    ├── workdir0-enwik8
      │        ├── check_point
      │            ├── model.pt
      │    ├── ...
      ```


   2. 导出onnx文件

      a. 使用run_enwik8_base.sh导出onnx文件

         运行run_enwik8_base.sh脚本
         ```
         chmod 777 ./run_enwik8_base.sh 
         ./run_enwik8_base.sh onnx --work_dir=workdir0-enwik8/check_point
         ```
         - 参数说明：
            - 第一个参数代表执行的模式
            - 第二个参数代表权重文件所在的路径
         运行成功后，在当前目录得到model.onnx文件

         注：此模型当前仅支持batch_size=1

      b. 简化模型

         对model.onnx使用onnx-simplifer工具进行简化，将模型中shape固定下来，以提升性能。

         ```
         python3 -m onnxsim model.onnx model_sim.onnx
         ```
         - 参数说明：
            - 第一个参数代表输入需要被简化的ONNX模型文件
            - 第二个参数代表输出简化后ONNX模型文件
         运行成功后，在当前目录得到model_sim.onnx文件
      
      c. 修改模型

         进入om_gener目录，执行以下命令安装改图工具
         ```
         cd om_gener
         pip3 install .
         cd ..
         ```
         对模型进行修改，执行modify_model.py脚本
         ```
         python3 modify_model.py model_sim.onnx
         ```
         运行成功后，在当前目录得到model_sim_new.onnx文件

   3. 使用ATC工具将ONNX模型转OM模型

      a. 确保device空闲
         ```
         npu-smi info
         ```
         ```
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

      b. 执行ATC命令

         执行atc.sh脚本，将.onnx文件转为离线推理模型文件.om文件

         ```
         bash atc.sh model_sim_new.onnx model_tsxl Ascend310P3
         ```
         - 参数说明：
            - 第一个参数代表输入的ONNX模型文件
            - 第二个参数代表输出的OM模型文件
            - 第三个参数代表芯片型号

         运行成功后，在当前目录得到model_tsxl.om模型文件


2. 模型推理

   利用ais_infer工具进行推理，ais_infer工具安装参考该[链接](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。由于该模型的特殊性，无法直接使用ais_infer.py文件进行推理，故调用aclruntime的接口进行推理验证。

   执行run_enwik8_base.sh脚本进行性能和精度的推理

   ```
   ./run_enwik8_base.sh om_eval --om_path=./model_tsxl.om  --work_dir=./workdir0-enwik8/check_point
   ```
   - 参数说明：
      - 第一个参数代表执行模式
      - 第二个参数代表推理的om模型
      - 第三个参数代表权重文件路径

   运行成功后，在屏幕上打印出性能和精度数据



# 推理结果<a name="ZH-CN_TOPIC_0000001172201573"></a>

| 模型 | 精度基准 | 310精度 | 310P精度 | 310性能 | 310P性能 | T4性能    | 310P/310 | 310P/T4 |
| ----------------- | ----------- | ---------------| --------------- | ---------- | ---------- |---------- |---------- |---------- |
| TransformerXl(bs1)  | 1.96636 | 1.96640 | 1.96661 | 18.7183 | 92.7141 | 94.0111 | 4.9531 | 0.9862 |

备注：

- 该模型只支持batch_size=1的情况;

- 性能单位:fps，精度指标为bpc;