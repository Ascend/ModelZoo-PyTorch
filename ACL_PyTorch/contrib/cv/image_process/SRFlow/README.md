# SRFlow模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

SRFlow是一种基于归一化流的超分辨率方法，具备比GAN更强的脑补能力，能够基于低分辨率输入学习输出的条件分布。


- 参考实现：

  ```
  url=https://github.com/andreas128/SRFlow
  branch=master
  commit_id=8d91d81a2aec17e7739c5822f3a5462c908744f8
  model_name=SRFlow
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 256 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型     | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | RGB_FP32 | batchsize x 3 x 2048 x 2048  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| onnx                                                      | 1.8.0   | -                                                            |
| TorchVision                                               | 0.9.0   | -                                                            |
| onnx-simplifier                                            | 0.3.6   | -                                                            |
| Pillow                                            | 8.2.0   | -                                                            |
| Opencv-python                                            | 4.4.0.46   | -                                                            |
| pyyaml                                                      | 5.3.1   | -                                                            |
| scikit-image                                                 | 5.3.1   | -                                                            |
| pyyaml                                                      | 0.17.2   | -                                                            |
| natsort                                                      | 7.0.1   | -                                                            |
| protobuf                                                 | 3.19.0   | -                                                            |
| decorator                                                   | 5.1.1   | -                                                            |
| sympy                                                      | 1.10.1   | -                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/andreas128/SRFlow -b master
   cd SRFlow
   git reset 8d91d81a2aec17e7739c5822f3a5462c908744f8 --hard
   patch -p1 < ../srflow.diff
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集，并将数据集移动至SRFlow路径下。
   ```
   wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip
   unzip datasets.zip
   rm datasets.zip
   ```
   解压后数据集目录结构如下所示：
   ```
   ├── datasets
       ├── div2k-validation-modcrop8-gt
       ├── div2k-validation-modcrop8-x8
   ```
2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行srflow_preprocess.py脚本，完成预处理。

   ```
   python3.7 srflow_preprocess.py -s ./SRFlow/datasets/div2k-validation-modcrop8-x8 -d ./prep_data
   ```

   将输入数据进行固定shape并将处理后的bin文件保存至./pre_data目录下。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件，并将权重文件SRFlow_DF2K_8X.pth移动至SRFlow路径下。

       ```
       wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow/pretrained_models.zip
       unzip pretrained_models.zip
       rm pretrained_models.zip
       ```

   2. 导出onnx文件。

      1. 使用srflow_pth2onnx.py导出onnx文件。

         运行srflow_pth2onnx.py脚本导出指定batch size为1的onnx模型，模型不支持动态batch。

         ```
         python3.7 srflow_pth2onnx.py  --pth ./SRFlow/SRFlow_DF2K_8X.pth --onnx srflow_df2k_x8_bs1.onnx
         ```

         获得srflow_df2k_x8_bs1.onnx文件。

      2. 优化ONNX文件。
         执行命令，对onnx模型进行优化。

         ```
         python3.7 -m onnxsim srflow_df2k_x8_bs1.onnx srflow_df2k_x8_bs1_sim.onnx 0 --input-shape "1,3,256,256"
         ```

         得到srflow_df2k_x8_bs1sim.onnx文件，onnxsim的三个参数分别代表输入的onnx模型、简化后输出的onnx模型与onnxsim检查次数。

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
         #该设备芯片名为Ascend310P3
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
          atc --framework=5 --model=srflow_df2k_x8_bs1_sim.onnx --output=srflow_df2k_x8_bs1 --input_format=NCHW --input_shape="input_1:1,3,256,256" --log=debug --soc_version=Ascend${chip_name} --fusion_switch_file=./fusion_switch.cfg
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --fusion_switch_file：融合配置文件。

           运行成功后生成srflow_df2k_x8_bs1.om模型文件。



2. 开始推理验证。

   a.  使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]，推理工具安装：
      ```
        git clone https://gitee.com/ascend/tools.git
        cd tools/ais-bench_workload/tool/ais_infer/
        pip3 wheel ./backend/ -v
        pip3 wheel ./ -v
        pip3 install ./aclruntime-{version}-cp37-cp37m-linux_xxx.whl
        pip3 install ./ais_bench-{version}-py3-none-any.whl
      ```

   b.  执行推理。

      ```
        # 创建result文件夹，存放推理结果文件
        rm -rf result
        mkdir result
        
        # 推理，执行ais_bench工具请选择与运行环境架构相同的命令。
        python -m ais_bench --device 0 --batchsize 1 --model ./srflow_df2k_x8_bs1.om --input ./prep_data/bin/ --output ./result/
      ```
      -m ais_bench为ais_bench工具脚本路径。
      -   参数说明：
   
           -   model：om模型类型。
           -   input：预处理后的 bin 文件路径。
           -   output：输出文件存放路径
           -   device：NPU设备编号。
    	...
   
      推理后的输出默认在当前目录result下。
   
      >**说明：** 
      >执行ais_bench工具请选择与运行环境架构相同的命令。

   c.  精度验证。

      调用srflow_postprocess.py脚本推理结果与Ground Truth比对，可以获得PSNR数据。
   
      ```
       python srflow_postprocess.py --hr ./SRFlow/datasets/div2k-validation-modcrop8-gt/ --binres ./result/2022_xx_xx-xx_xx_xx/  --save ./result/2022_xx_xx-xx_xx_xx/save/
      ```
   
      ./result/2022_xx_xx-xx_xx_xx/sumary.json 中的 2022_xx_xx-xx_xx_xx 为 ais_infer 自动生成的目录名。  
      参数解释：
   
      hr：对应输入低分变率图片的高分辨率图片路径
   
      binres：推理的输出路径
   
      save：代表对推理结果进行处理并将处理结果保存为图片的路径


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

 **精度评测结果：**   

|    模型    | 官网pth精度 | 310离线推理精度 | 310P3离线推理精度 |
| :--------: | :---------: | :-------------: | :---------------: |
| SRFlow bs1 | psnr:23.05  |   psnr:23.063   |    psnr:23.041    |

**性能评测结果：**

|    模型    |  310性能   | 310P3性能  |
| :--------: | :--------: | :--------: |
| SRFlow bs1 | 0.53768fps | 0.72827fps |