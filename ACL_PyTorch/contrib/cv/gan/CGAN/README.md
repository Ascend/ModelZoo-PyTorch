# CGAN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

CGAN(条件生成对抗网络,Conditional Generative Adversarial Nets）是生成对抗网络(GAN)的条件版本.可以通过简单地向模型输入数据来构建.在无条件的生成模型中，对于生成的数据没有模式方面的控制，很有可能造成模式坍塌．而条件生成对抗网络的思想就是通过输入条件数据，来约束模型生成的数据的模式．输入的条件数据可以是类别标签，也可以是训练数据的一部分，又甚至是不同模式的数据．CGAN的中心思想是希望可以控制 GAN 生成的图片，而不是单纯的随机生成图片。


- 参考实现：

  ```
  url=https://github.com/znxlwm/pytorch-generative-model-collections
  branch=master
  commit_id=0d183bb5ea2fbe069e1c6806c4a9a1fd8e81656f
  model_name=CGAN
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

  | 输入数据 | 数据类型 | 大小 | 数据排布格式 |
  | -------- | -------- | --------- | ------------ |
  | input    | FP32 | 100 x 72  | HW         |


- 输出数据

  | 输出数据 | 数据类型 |大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32 |100 x 3 x 28 x 28  | NCHW           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

**表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


   **表 2**  环境依赖表


```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 生成器一次只能生成一张图，由于模型输入是两维的，不是常用的NCHW格式，input_format采用ND形式
atc --framework=5 --model=CGAN_sim.onnx --output=CGAN_bs1 --input_format=ND --output_type=FP32 --input_shape="image:100,72" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
```
=======
| 依赖                                                         | 版本    |
| ------------------------------------------------------------ | ------- |
| torchvision                                                  | 0.6.0   |
| Pillow                                                       | 8.4.0   |
| six                                                          | 1.16.0  |
| onnx                                                         | 1.10.2  |
| protobuf                                                     | 3.20.1  |
| onnxruntime                                                  | 1.9.0   |
| scipy                                                        | 1.7.1   |
| numpy                                                        | 1.21.2  |
| onnx-simplifier                                              | 0.3.6   |
| imageio                                                      | 2.9.0   |
| matplotlib                                                   | 3.4.3   |
| decorator                                                    | 5.1.1   |
| sympy                                                        | 1.10.1  |
| aclruntime                                                   | 0.0.1   |
| tqdm                                                         | 4.64.1  |
>>>>>>> 添加CGAN 310P修改


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）。

   请用户自行获取“CGAN_G.pth”和源码包。
   源码目录为：

```
|CGAN--test
|     |   |--pth2om.sh
|     |   |--eval_acc_perf.sh
|     |   |--perf_t4.sh
|     |--util.py
|     |--CGAN.py
|     |--gen_dataset_info.py
|     |--CGAN_pth2onnx.py
|     |--CGAN_preprocess.py
|     |--CGAN_postprocess.py
|     |--requirements.txt
|     |--LICENCE
|     |--modelzoo_level.txt
|     |--README.md
```

2. 安装依赖。

   ```
   cd ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/gan/CGAN
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

      本模型使用的是随机数作为生成网络的输入，运行“CGAN_preprocess.py”会自动生成随机数并转化为二进制文件保存在“./prep_dataset”目录下，本模型将使用随机数测试模型性能

 

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行“CGAN_preprocess.py”脚本，完成预处理。

   ```
   python3.7  CGAN_preprocess.py --save_path ./prep_dataset
   ```
   
   - 参数说明：

      -   --save_path：输出的二进制文件（.bin）所在路径。
   
   在当前目录下生成“prep_dataset”二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。


```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
=======
   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
>>>>>>> 添加CGAN 310P修改

   1. 获取权重文件。

      自行获取权重文件：“CGAN_G.pth”。


   2. 导出onnx文件。

      1. 使用“CGAN_G.pth”导出onnx文件。

         运行“CGAN_pth2onnx.py”脚本。

         ```
         python3.7 CGAN_pth2onnx.py --pth_path ./CGAN_G.pth --onnx_path ./CGAN.onnx
         ```

         获得“CGAN.onnx”文件。,本模型只支持bs1,batch_size不做改变。

      2. 使用onnxsim优化onnx模型

         优化模型。
         ```
         python3.7 -m onnxsim --input-shape="100,72" CGAN.onnx CGAN_sim.onnx
         ```

         获得“CGAN_sim.onnx”文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
         
      2. 执行ATC命令。
         ```
         atc --framework=5 --model=CGAN_sim.onnx --output=CGAN_bs1 --input_format=ND --output_type=FP32 --input_shape="image:100,72" --log=debug --soc_version=${chip_name}
         ```
         ${chip_name}可通过 npu-smi info指令查看。
        ~~~
        +--------------------------------------------------------------------------------------------+
        | npu-smi 22.0.0                       Version: 22.0.2                                       |
        +-------------------+-----------------+------------------------------------------------------+
        | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
        | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
        +===================+=================+======================================================+
        | 0       310P3     | OK              | 16.7         57                0    / 0              |
        | 0       0         | 0000:5E:00.0    | 0            932  / 21534                            |
        +===================+=================+======================================================+
        ~~~
         本报告测试时为Ascend310P3。
         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --output\_type：输出数据的格式
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成“CGAN_bs1.om”模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  

   b.  执行推理。

      ```
        python3.7 -m ais_bench  --device 0  --model CGAN_bs1.om --output ./ --outfmt BIN --loop 5 --input prep_dataset
      ```

      -   参数说明：

           -   --device：指定设备。
           -   --model：om文件路径。
           -   --output：输出结果路径。
           -   --outfmt：输出类型。
           -   --loop： 命令循环次数。
           -   --input：预处理后的输入数据。
		

      推理后的输出默认在当前目录下。根据推理时间生成结果文件夹。例如：/2022_09_04-17_57_51

      >**说明：** 
      >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。

      将ais_bench推理获得的BIN输出结果进行后处理，保存为图片。

      ```
      python3.7 CGAN_postprocess.py --bin_out_path ./2022_09_04-17_57_51 --save_path ./result
      ```
      -   参数说明：
    
            -   --bin_out_path：推理结果二进制文件所在目录，’/2022_09_04-17_57_51‘是离线推理时根据时间自动生成的目录，请根据实际情况改变。
            -   --save_path:保存后处理产生的图片的目录     
      
      将位于/result目录下的结果图片与用户自行获取的开源精度图片进行比较，结果图片可以正常生成数字，与pth模型生成的图片大致相同即符合精度要求。


   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证om模型的性能，因为CGAN模型只支持bs1,所以不需要考虑batch_size的变化。参考命令如下：

      ```
      python3.7 -m ais_bench  --device 0  --model CGAN_bs1.om --output ./ --outfmt BIN --loop 5 --input prep_dataset
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

   调用ACL接口推理计算，性能参考下列数据。

   1.精度对比

   将结果图片与README.assets/result.png进行比较，结果图片可以生成数字，与开源精度图片大致相同即符合精度要求。

   2.性能对比
   | Throughput | 310     | 310P    | T4     | 310P/310    | 310P/T4     |
   | ---------- | ------- | ------- | ------ | ----------- | ----------- |
   |        bs1    | 1602.5641 | 1935.7336 | 2538.1070 | 1.208 | 0.763 |