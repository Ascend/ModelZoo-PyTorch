# BigGAN模型-推理指导


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

BigGAN是用于图像生成的大规模生成对抗网络。相较于先前的生成对抗网络。BigGAN增大了网络规模，具体来说网络具有更大的参数量，同时训练过程中采用更大的batch size。为获得更好的生成图像，BigGAN引入了正交正则化和数据截断。正交正则化：BigGAN将正交正则化引入生成器网络中并进行调整，使模型对于经过截断处理的数据具有一定适应性，降低生成图像中伪影的产生；数据截断：对BigGAN的噪声输入进行截断处理，当随机采样获得的数大于给定阈值则重新采样，阈值的设定将影响生成的图像多样性和真实性。通过这些技巧的使用，BigGAN大幅度提升了基于类别的合成图像的质量。


- 参考实现：

  ```
  url=https://github.com/ajbrock/BigGAN-PyTorch
  commit_id=98459431a5d618d644d54cd1e9fceb1e5045648d
  code_path=https://gitee.com/ascend/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/gan
  model_name=BigGAN
  ```
  
- 参考论文：

  [Brock A, Donahue J, Simonyan K. Large scale GAN training for high fidelity natural image synthesis[J]. arXiv preprint arXiv:1809.11096, 2018.](https://arxiv.org/abs/1809.11096)



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | noise    | FLOAT32  | batchsize x1x 20  | ND           |
  | label    | FLOAT32  | batchsize x5x 148 | ND           |


- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------------------- | ------------ |
  | image    | FLOAT32  | batchsize x3x 128x128 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本   | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.4 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.1  | -                                                            |
  | Python                                                       | 3.7.5  | -                                                            |
  | PyTorch                                                      | 1.6.0  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \      | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/ajbrock/BigGAN-PyTorch.git
   mv biggan.patch BigGAN-PyTorch
   cd BigGAN-PyTorch
   dos2unix biggan.patch
   git apply biggan.patch
   cp BigGAN.py ..
   cp layers.py ..
   cp inception_utils.py ..
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   BigGAN模型的输入数据是由噪声数据和标签数据组成，其中噪声数据是由均值为0，方差为1的正态分布中采样，标签数据是由0至类别总数中随机采样一个整数.。针对不同的batch size需要生成不同的输入数据。

2. 数据预处理。

   执行输入数据的生成脚本biggan_preprocess.py，生成模型输入的bin文件

   ```
   python3 biggan_preprocess.py --num-inputs 50000
   ```
   
     注：运行后生成“prep_label”和“prep_noise”文件夹和gen_y.npy文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       模型使用BigGAN预训练pth权重文件[G_ema.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/BigGan/PTH/G_ema.pth) 和Inception_v3预训练pth权重文件[inception_v3_google.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/BigGan/PTH/inception_v3_google.pth) 。

       注： 其中下载的权重文件名为inception_v3_google.pth，此模型权重用于IS评价指标的计算，若仅进行图像生成，无需下载此权重文件 

   2. 导出onnx文件。

      1. 使用 G_ema.pth导出onnx文件。

         运行biggan_pth2onnx.py脚本。

         ```
         python3 biggan_pth2onnx.py --source "./G_ema.pth" --target "./biggan.onnx"
         ```

         获得biggan.onnx文件。

      2. 执行clip_edit.py脚本，通过"input-model"和"output-model"参数指定输入和输出的onnx模型，默认输入输出均为"./biggan.onnx"

         ```
         python3 clip_edit.py --input-model biggan.onnx --output-model new_biggan.onnx
         ```

         注： 执行clip_edit.py目的在于初始化onnx模型中Clip节点中的"max"输入，便于后续onnx模型的简化 。
         
      3. 优化ONNX文件。

          使用onnx-simplifier简化onnx模型 ， 生成batch size为1的简化onnx模型，对应的命令为： 

         ```
         python3 -m onnxsim './new_biggan.onnx' './biggan_sim_bs{batchsize}.onnx' --input-shape noise:{batchsize},1,20 label:{batchsize},5,148
         ```

         获得biggan_sim_bs{batchsize}.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

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

         ```
         atc --framework=5 --model=./biggan_sim_bs{batchsize}.onnx --output=./biggan_sim_bs{batchsize} --input_format=ND --input_shape="noise:{batchsize},1,20;label:{batchsize},5,148" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
      
      运行成功后生成biggan_sim_bs{batchsize}.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model biggan_sim_bs1.om --input ./prep_noise,./prep_label  --output ./result --output_dirname bs1 --outfmt BIN --batchsize 1
        ```
        
        + 参数说明：
           -   --model：输入的om文件。
           -    --input：输入的bin数据文件。
           -   --output：推理数据输出路径。
           -   --outfmt：输出数据的格式。
           -   --batchsize：模型batch size。
        
        推理后的输出默认在当前目录result下。
   
   3. 精度验证。
   
      
      1. 模型后处理将离线推理得到的bin文件转换为jpg图像文件，并将原始输出保存至npz文件中，用于精度数据的获取 。
      
         ```
         python3 biggan_postprocess.py --result-path ./result/bs1 --save-path ./postprocess_img --batch-size 1 --save-img --save-npz
         ```
      
         + 参数说明：
            -   --save-img：启用后将bin文件转换为jpg图像文件。
            -   --save-path：后处理得到的图像文件的存储路径。 
            -   --result-path：离线推理的原始输出路径。
            -   --batch-size：输入参数。
               
         注：每个bs都需要进行数据后处理。 
      
      2. 调用biggan_eval_acc.py脚本与ImageNet数据集采[I128_inception_moments.npz](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/BigGan/PTH/I128_inception_moments.npz)进行计算可以获得FID数据，借助预训练的Inception_v3模型，能够计算IS数据，结果保存在 biggan_acc_eval_bs1.log中。
      
         ```
         python3 biggan_eval_acc.py --num-inception-images 50000 --batch-size 1 --dataset 'I128'
         ```
      
          + 参数说明：
             -   --num-inception-images：用于精度计算的输出数量。
             -   --dataset：指定计算FID时对比的分布，I128表示ImageNet。
             -   --batch-size：输入参数。
      
         注：每个bs都需要进行数据后处理，且BigGAN每次运行数据来自随机采样，IS和FID数据存在波动。
      
         > **说明**
         > IS是生成图像的清晰度和多样性指标，其值越大说明越优
         > FID是生成图像集与真实图像集间的相似度指标，其值越小说明越优

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=biggan_sim_bs${batch_size}.om --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：输入的om文件。
        - --batchsize：模型batch size。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

精度：

| 模型               | IS               | FID     |
| ------------------ | ---------------- | ------- |
| pth模型推理结果    | 94.323+/-2.395   | 9.9532  |
| om模型离线推理结果 | 94.293 +/- 2.190 | 10.0190 |

性能：

| 芯片型号 | Batch Size   | 性能 |
| --------- | ---------------- | --------------- |
| 310P3 | 1 | 467.7174 |
| 310P3 | 4 | 550.944 |
| 310P3 | 8 | 463.0535 |
| 310P3 | 16 | 524.359 |
| 310P3 | 32 | 542.7187 |
| 310P3 | 64 | 489.6636 |