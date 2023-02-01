# SRCNN模型-推理指导


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

SRCNN是一种一种用于单一图像超级分辨率的深度学习方法，该方法学习了一个端到端、从低分辨率到高分辨率图像之间的映射。

- 参考实现：

  ```
  url=https://github.com/yjn870/SRCNN-pytorch
  branch=master
  commit_id=064dbaac09859f5fa1b35608ab90145e2d60828b
  model_name=SRCNN
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

  | 输入数据  | 数据类型  | 大小                      | 数据排布格式  |
  | -------- | -------- | ------------------------- | ------------ |
  | input.1  | FLOAT32  | batchsize x 1 x 256 x 256 | NCHW         |


- 输出数据

  | 输出数据  | 数据类型  | 大小                      | 数据排布格式  |
  | -------- | -------- | ------------------------- | ------------ |
  | 11       | FLOAT32  | batchsize x 1 x 256 x 256 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

   | 配套       | 版本     | 环境准备指导             |
   | ---------- | ------- | ----------------------- |
   | 固件与驱动  | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
   | CANN       | 6.0.0   | -                       |
   | Python     | 3.7.5   | -                       |
   | PyTorch    | 1.5.0   | -                       |  

- 该模型需要以下依赖   

  **表 2**  依赖列表

  | 依赖名称               | 版本                    |
  | --------------------- | ----------------------- |
  | torchvision           | 0.6.0                   |
  | onnx                  | 1.9.0                   |
  | numpy                 | 1.19.2                  |
  | Pillow                | 8.2.0                   |
  | opencv-python         | 4.5.2                   |

   > **说明：**
   >
   > X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3 install 包名 安装 
   >
   > Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3 install 包名 安装

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/yjn870/SRCNN-pytorch
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用[Set5官网](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)的5张验证集进行测试，图片存放在/root/dataset/set5下面。

2. 数据预处理，将原始数据集转换为模型输入的数据。

   使用 srcnn_preprocess.py 脚本进行数据预处理，脚本执行命令：

   ```
   python3 srcnn_preprocess.py -s /root/datasets/set5 -d ./prep_data
   ```

   预处理脚本会在./prep_data/png/下保存中心裁剪大小为256*256的预处理图片，并在缩放处理之后将bin文件保存至./prep_data/bin/下面。

   执行生成数据集信息脚本get_info.py，生成数据集信息文件：

   ```
    python3 get_info.py bin ./prep_data/bin ./srcnn_prep_bin.info 256 256
   ```

   第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载[SRCNN_x2预训练pth权重文件](https://www.dropbox.com/s/rxluu1y8ptjm4rn/srcnn_x2.pth?dl=0)。
       
   2. 导出onnx文件。

         使用 srcnn_pth2onnx.py 转换pth为onnx文件，在命令行运行如下指令：

         ```
         python3 srcnn_pth2onnx.py --pth srcnn_x2.pth --onnx srcnn_x2.onnx
         ```

         获得srcnn_x2.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
             +--------------------------------------------------------------------------------------------+
             | npu-smi 22.0.0                       Version: 22.0.2                                       |
             +-------------------+-----------------+------------------------------------------------------+
             | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
             | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
             +===================+=================+======================================================+
             | 0       310P3     | OK              | 17.0         56                0    / 0              |
             | 0       0         | 0000:AF:00.0    | 0            934  / 23054                            |
             +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```
         atc --framework=5 --model=srcnn_x2.onnx --output=srcnn_x2 --input_format=NCHW --input_shape="input.1:1,1,256,256" --log=debug --soc_version=Ascend310
         ```

         - 参数说明：

            - --model：为ONNX模型文件
            - --framework：5代表ONNX模型
            - --output：输出的OM模型
            - --input_format：输入数据的格式
            - --input_shape：输入数据的shape
            - --log：日志级别
            - --soc_version：处理器型号

           运行成功后生成srcnn_x2.om模型文件。

2. 开始推理验证。
   
   1. benchmark工具概述。

      benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN 5.0.1 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100191895)。

   2.  执行推理。

         ```
         ./benchmark -model_type=vision -device_id=0 -batch_size=1 -om_path=./srcnn_x2.om -input_text_path=./srcnn_prep_bin.info -input_width=256 -input_height=256 -output_binary=False -useDvpp=False
         ```
   
      输出结果默认保存在当前目录result/dumpOutput_device{0}，对应了Set5中每张图片的输出。

   3.  精度验证。

      1. 离线推理精度

         后处理输出每一张图片在经过SRCNN处理之后的PSRN的值，同时将处理后的图片保存在./result/save目录下。调用srcnn_postprocess.py来进行后处理，结果输出在控制台上。

         ```
         python3 srcnn_postprocess.py --res ./result/dumpOutput_device0/ --png_src ./prep_data/png --bin_src ./prep_data/bin --save ./result/save
         ```

         第一行为处理的图片名，第二行为PSNR的计算结果：

         ```
         woman_256.png PSNR: 36.79

         bird_256.png PSNR: 37.61

         head_256.png PSNR: 41.38

         baby_256.png PSNR: 36.31

         butterfly_256.png PSNR: 29.59

         total PSNR: 36.33
         ```

         上述结果中的total psnr是通过对于所有psnr结果取平均值得到的。

      2. 精度对比

         根据该模型github上代码仓所提供的测试脚本，运行pth文件，得到预训练文件在官方代码仓库上对于Set5的验证集的PSNR输出值。

         ```
         cd SRCNN-pytorch
         echo "woman "
         python3 test.py --weights-file "../srcnn_x2.pth" --image-file "../prep_data/png/woman_256.png" --scale 2
         echo ""
         echo "bird "
         python3 test.py --weights-file "../srcnn_x2.pth" --image-file "../prep_data/png/bird_256.png" --scale 2
         echo ""
         echo "head "
         python3 test.py --weights-file "../srcnn_x2.pth" --image-file "../prep_data/png/head_256.png" --scale 2
         echo ""
         echo "baby "
         python3 test.py --weights-file "../srcnn_x2.pth" --image-file "../prep_data/png/baby_256.png" --scale 2
         echo ""
         echo "butterfly "
         python3 test.py --weights-file "../srcnn_x2.pth" --image-file "../prep_data/png/butterfly_256.png" --scale 2
         echo ""
         ```   

   4. 性能验证。

      1. npu性能数据

         benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。

         benchmark工具作纯推理时使用的命令参考如下：

         ```
         ./benchmark -round=20 -om_path=./srcnn_x2.om -device_id=0 -batch_size=1
         ```
      
      2. T4性能数据

         在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

         ```
         trtexec --onnx=srcnn_x2.onnx --fp16 --shapes=image:1x1x256x256 --threads
         ```

         gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度数据参考如下：

|         |   原github测试脚本PSNR   |   om模型PSNR   |
| :-----: | :------: | :------: |
|  woman  | 36.89 | 36.79  |
| bird |  37.74  |  37.61  |
| head |  41.24  |  41.38  |
| baby |  36.52  |  36.31  |
| butterfly |  29.55  |  29.59  |

将得到的om离线模型推理精度与该模型github代码仓上公布的精度对比，精度下降均在1%范围之内。同时，total psnr为36.33dB，对比开源仓库的36.65dB而言，下降精度0.9%也在1%以内，故精度达标。

性能数据参考如下：

benchmark工具在整个数据集上的运行结果如下：
   ```
    [e2e] throughputRate: 3.74691, latency: 1334.43
    [data read] throughputRate: 2857.14, moduleLatency: 0.35
    [preprocess] throughputRate: 312.48, moduleLatency: 3.2002
    [infer] throughputRate: 285.682, Interface throughputRate: 348.038, moduleLatency: 3.4136
    [post] throughputRate: 50.2841, moduleLatency: 19.887
   ```
Interface throughputRate: 348.038 * 4 = 1392.152 即是batch1 310单卡吞吐率


trtexec工具在整个数据集上的运行结果如下：
   ```
    [07/18/2021-15:30:00] [I] GPU Compute
    [07/18/2021-15:30:00] [I] min: 0.796875 ms
    [07/18/2021-15:30:00] [I] max: 3.98999 ms
    [07/18/2021-15:30:00] [I] mean: 0.859765 ms
    [07/18/2021-15:30:00] [I] median: 0.853516 ms
    [07/18/2021-15:30:00] [I] percentile: 0.987305 ms at 99%
    [07/18/2021-15:30:00] [I] total compute time: 2.98597 s
   ```
batch1 t4单卡吞吐率：1000/(0.859765/1)=1163.108fps