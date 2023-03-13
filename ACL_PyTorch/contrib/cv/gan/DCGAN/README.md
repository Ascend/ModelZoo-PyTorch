# DCGAN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DCGAN是生成对抗网络GAN中一种常见的模型结构。其中的生成器和判别器都是神经网络模型。DCGAN是GAN的一个变体，DCGAN就是将CNN和原始的GAN结合到一起，生成网络和鉴别网络都运用到了深度卷积神经网络。DCGAN提高了基础GAN的稳定性和生成结果质量。


- 参考实现：

  ```
  url=https://github.com/eriklindernoren/PyTorch-GAN.git
  branch=master
  commit_id=36d3c77e5ff20ebe0aeefd322326a134a279b93e
  model_name=DCGAN
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone https://github.com/eriklindernoren/PyTorch-GAN.git   # 克隆仓库的代码
  cd PyTorch-GAN                                                 # 切换到模型的代码仓目录
  git reset 36d3c77e5ff20ebe0aeefd322326a134a279b93e --hard      # 切换到对应分支
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据
  | 输入数据 | 数据类型   | 大小                      | 数据排布格式 |
  | ------- | --------  | ------------------------- | ------------ |
  | input   | noise_FP16 | batchsize x 100 x 1 x 1 | NCHW         |


- 输出数据
  | 输入数据 | 数据类型 | 大小                     | 数据排布格式 |
  | ------- | -------- | ----------------------- | ----------- |
  | output | GREY_FP16 | batchsize x 1 x 28 x 28 | NCHW        |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

上传源码包到服务器任意目录并解压

```
├── checkpoint-amp-epoch_200.pth    //DCGAN pth模型
├── ReadMe.md
├── dcgan.patch                     //patch文件，修改开源代码仓代码  
├── dcgan_acc_eval.py               //pth与om各自生成结果的精度验证脚本 
├── dcgan_postprocess.py            //将离线推理得到的bin文件可视化为PNG图像
├── dcgan_preprocess.py             //采样生成输入DCGAN模型的随机噪声 
├── dcgan_pth_result.py             //基于pth文件在cpu上利用预生成的随机噪声生成结果，用以之后的精度对比
├── dcgan_pth2onnx.py               //pth转onnx的python脚本  
└── requirements.txt                //环境依赖文件
```

1. 获取源码。
  ```
  git clone https://github.com/eriklindernoren/PyTorch-GAN.git
  cd PyTorch-GAN
  git reset 36d3c77e5ff20ebe0aeefd322326a134a279b93e --hard
  mv ../dcgan.patch ./
  git apply dcgan.patch
  cd ..
  ```

2. 安装依赖。

  ```
  pip3 install -r requirements.txt
  ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    DCGAN的模型输入是随机噪声，无需下载原始数据集。

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

    数据预处理将从正态分布中采样随机噪声作为模型的输入。默认设置下噪声样本数为8192。

    执行`dcgan_preprocess.py`脚本，完成预处理。
    ~~~shell
    python3 dcgan_preprocess.py ./prep_dataset
    ~~~
    - 参数说明：
      -   `./prep_dataset` 输出的二进制文件（.bin）所在路径

    运行成功后会生成名为`prep_dataset`目录，其中包含裁剪后的图像，并以bin的格式存储。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取DCGAN预训练权重文件:[checkpoint-amp-epoch_200.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/DCGAN/PTH/checkpoint-amp-epoch_200.pth)



   2. 导出onnx文件。

      1. 使用`checkpoint-amp-epoch_200.pth`导出onnx文件。

          运行`dcgan_pth2onnx.py`脚本。
          ```shell
          python3 dcgan_pth2onnx.py ./checkpoint-amp-epoch_200.pth ./dcgan.onnx
          ```
          - 参数说明：
            -   `./checkpoint-amp-epoch_200.pth`: 为pth模型路径
            -   `./dcgan.onnx`: 为导出的onnx模型路径
        
          最终获得`dcgan.onnx`文件。

      2. 使用onnx-simplifier工具简化原始onnx文件。
          ```shell
          python3.7 -m onnxsim --input-shape=1,100,1,1 dcgan.onnx dcgan_sim_bs{batch_size}.onnx
          ```
          最终获得`dcgan_sim_bs{batch_size}.onnx`文件。


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
          ```shell
          atc --framework=5 --model=./dcgan_sim_bs{batch_size}.onnx --output=dcgan_sim_bs{batch_size} --input_format=NCHW --input_shape="noise:{batch_size},100,1,1" --log=error --soc_version=Ascend${chip_name}
          ```

          - 参数说明：
            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input\_format：输入数据的格式。
            -   --input\_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc\_version：处理器型号。
          
           运行成功后生成`dcgan_sim_bs{batch_size}.om`模型文件。


2. 开始推理验证。

   a.  安装ais_bench推理工具。

    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   b.  执行推理。
      ```shell
      python3 -m ais_bench --model=./dcgan_sim_bs{batch_size}.om --batchsize=${batchsize} --output=./result --input=./prep_dataset --loop=2000
      ```
    - 参数说明：
      - --model：om模型路径
      - --batchsize：batchisize大小
      - --output: 输出路径
      - --input：输入bin文件目录
      - --loop: 循环测试轮数
 
    推理后的输出默认在当前目录result下。

   c.  精度验证。

    1. 调用“dcgan_pth_result.py”脚本生成pth模型在cpu上的结果。输入数据是之前生成的“prep_dataset”数据集。
        ```shell
        python3 dcgan_pth_result.py --checkpoint_path=./checkpoint-amp-epoch_200.pth --dataset_path=./prep_dataset --save_path=./pth_result
        ```
        - 参数说明：
          - --checkpoint_path：权重文件所在路径
          - --dataset_path：数据集路径
          - --save_path: 输出路径
    
    2. 调用“dcgan_acc_eval.py”脚本对比cpu生成结果与npu生成结果。执行以下命令生成batchsize1的精度对比结果。
        ```shell
        python3 dcgan_acc_eval.py --pth_result_path=./pth_result/ --om_result_path=./result/dumpOutput_device0/ --log_save_name=dcgan_acc_eval_bs{batch_size}.log
        ```
        - 参数说明：
          - --pth_result_path：pth文件生成结果路径
          - --om_result_path：推理结果路径
          - --log_save_name: 精度结果log文件

   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
      ```shell
      python3 -m ais_bench --model=./dcgan_sim_bs{batch_size}.om --batchsize=${batchsize} --output=./result --input=./prep_dataset
      ```

    - 参数说明：
      - --model：om模型路径
      - --batchsize：batchisize大小
      - --output: 输出路径
      - --input：输入bin文件目录



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。
  1. 性能对比

| Throughput | 310        | 310P      | T4         | 310P/310    | 310P/T4 |
| ---------- | ---------  | --------- | ---------- | ----------- | ------- |
| bs1        | 10773.603  | 7573.250  | 9330.62    | 0.70        | 0.81    |
| bs4        | 35709.304  | 34988.598 | 23644.02   | 0.98        | 1.48    |
| bs8        | 56937.918  | 55657.639 | 18339.4121 | 0.98        | 3.03    |
| bs16       | 81521.969  | 83132.896 | 23854.8907 | 1.02        | 3.48    |
| bs32       | 99208.283  | 108781.75 | 26682.0088 | 1.10        | 4.08    |
|            |            |           |            |             |         |
| 最优batch  | 99208.283  | 108781.75 | 26682.01   | 1.10        | 4.08    | 
  
  2. 精度对比

由于模型是GAN模型适用于手写数字生成的任务，因此输出没有基准精度，此处通过比较cpu生成结果与npu生成结果，并计算相似性。

精度数据为：100%

|模型 | batchsize |精度(mean)|精度(cosine)|精度(acc)|
| --- | --------- | ------- | ---------- | ------- |
|DCGAN| bs1       |0.0004   |1.0         |100.0%   |
|DCGAN| bs4       |0.0004   |1.0         |100.0%   |
|DCGAN| bs8       |0.0004   |1.0         |100.0%   |
|DCGAN| bs16      |0.0004   |1.0         |100.0%   |
|DCGAN| bs32       |0.0004   |1.0         |100.0%   |

log:
```shell
noise_0000_0.bin : mean ==> 0.0004 , cos ==> 1.0000
noise_0001_0.bin : mean ==> 0.0004 , cos ==> 1.0000
noise_0002_0.bin : mean ==> 0.0004 , cos ==> 1.0000
noise_0003_0.bin : mean ==> 0.0004 , cos ==> 1.0000
noise_0004_0.bin : mean ==> 0.0005 , cos ==> 1.0000
noise_0005_0.bin : mean ==> 0.0004 , cos ==> 1.0000
...
...
...
noise_8186_0.bin : mean ==> 0.0004 , cos ==> 1.0000
noise_8187_0.bin : mean ==> 0.0004 , cos ==> 1.0000
noise_8188_0.bin : mean ==> 0.0004 , cos ==> 1.0000
noise_8189_0.bin : mean ==> 0.0004 , cos ==> 1.0000
noise_8190_0.bin : mean ==> 0.0005 , cos ==> 1.0000
noise_8191_0.bin : mean ==> 0.0004 , cos ==> 1.0000
mean : 0.0004, cosine : 1.0000, acc : 100.00%
```
