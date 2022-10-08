# 3D_ResNet_ID0421模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

3D-CNN网络在传统卷积网络的基础上引入了3D卷积，使得模型能够提取数据的时间维度和空间维度特征，从而能够完成更复杂的图像识别或者动作识别任务。3D-ResNets将经典的残差网络结构Resnets与3D卷积结合，在动作识别领域达到了SOA水平。


- 参考实现：

  ```
  url=https://github.com/kenshohara/3D-ResNets-PyTorch
  branch=master
  commit_id=f399b376ca555f0ff925d77517313164c66504f9
  model_name=3d_resnets
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

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32| 10 x 3 x 16 x 112 x 112 | ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 10 x 51 | FLOAT32  | ND           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| onnx	| 1.7.0	| 1.7.0 |
| Torch	| 1.8.0	| 1.5.0 |
| TorchVision	| 0.9.0	| None | 
| numpy	| 1.21.2	| None | 
| Pillow	| 8.3.0	| None | 
| scikit-image	| 0.16.2	| None | 
| Opencv-python	| 4.6.0.66	| None |    

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型基于hmdb51数据集训练和推理，hmdb51是一个轻量的动作识别数据汇集，包含51种动作的短视频。hmdb51数据集的获取及处理参考Link(https://github.com/kenshohara/3D-ResNets-PyTorch)中Preparation和HMDB-51小节，处理后的数据格式为从视频帧中提取的jpg图片和标签json文件。

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将图片数据转换为模型输入的二进制数据，将原始数据（.jpg）转化为二进制文件（.bin）。执行3D-ResNets_postprocess.py脚本。

   ```
   python3 3D-ResNets_preprocess.py --video_path=hmdb51 --annotation_path=hmdb51_1.json --output_path=Binary_hmdb51 --dataset=hmdb51 --inference_batch_size=1
   ```
         - 参数说明：  
       
           -   --video_path：jpg数据最上层目录。
           -   --annotation_path：数据集信息路径。
           -   --output_path：输出目录。
           -   --dataset：数据集类型，默认hmdb51。
           -   --inference_batch_size：推理batch_size。
    运行完预处理脚本会在当前目录输出hmdb51.info文件和Binary_hmdb51二进制文件夹，包含视频片段名字和长度信息，用于后处理。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       − 从源码包中获取save_700.pth文件

   2. 导出onnx文件。

      1. 将模型权重文件.pt转换为.onnx文件。
         − 下载代码仓。

         ```
           git clone https://github.com/kenshohara/3D-ResNets-PyTorch.git
         ```

         − 将代码仓上传至服务器。修改主目录models/resnet.py文件中代码：

         ```
           self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
           改为
           self.avgpool = nn.AvgPool3d((1,4,4))
         ```
         − 进入代码仓目录并将save_700.pth与3D-ResNets_pth2onnx.py移到github项目主目录下。
         − 进入主目录，执行3D-ResNets_pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令。
         ```
           python3 3D-ResNets_pth2onnx.py --root_path=./ --video_path=hmdb51 --annotation_path=hmdb51_1.json --result_path=result --dataset=hmdb51 --model=resnet --model_depth=50 --n_classes=51 --resume_path=save_700.pth
         ```
         − 需用户创建result文件夹。
         − 对应参数信息可在github项目主目录中opts.py中查看。
         − 运行成功后，在当前目录生成3D-Resnets.onnx模型文件。然后将生成onnx文件移到ModelZoo源码包中。此模型当前仅支持batch_size=10。


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

         ```
         atc --model=3D-ResNets_sim.onnx --framework=5 --output=output_3D-ResNets --input_format=NCHW --input_shape="input:10,3,16,112,112" --log=info --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
           运行成功后生成<u>***3D-Resnets.om***</u>模型文件。



2. 开始推理验证。

a.  使用ais-infer工具进行推理。

   执行命令增加工具可执行权限，并根据OS架构选择工具

   ```
   chmod u+x 
   ```

b.  执行推理。

    ```
     python3 ais_infer.py --model 3D-Resnets.om --input Binary_hmdb51 --output result --batchsize=10
    ```
    
    -   参数说明：
    
        -   model：模型地址。
        -   input：预处理完的数据集文件夹。
        -   output：推理结果保存地址。默认会建立日期+时间的子文件夹保存输出结果，如果指定output_dirname将保存到output_dirname的子文件夹下。
        -   batchsize：模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的  batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。  
    
        推理后的输出默认在当前目录result下。
    
        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

c.  精度验证。

    - 运行后处理脚本3D-ResNets\_postprocess.py将推理结果处理成json文件。
    
    ```
     python3 3D-ResNets_postprocess.py out/20210408_084607/ 1
    ```
    
    第一个参数out/20210408_084607/为推理时自动生成的output目录，具体名称根据时间变化，请修改为实际名称。第二个参数是选择统计精度的topK的K值，如1表示统计top 1精度。运行成功后生成val.json文件。
    
    - 运行eval_accuracy.py脚本与数据集标签hmdb51_1.json比对，可以获得Accuracy数据。
    ```
     python3 eval_accuracy.py hmdb51_1.json val.json --subset validation -k 1 --ignore
    ```
    val.json 是后处理输出的json文件，subset选择评测的子集，k为统计精度topK的K值，--ignore用于忽略缺失数据。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310          | bs10                | hmdb51  | 0.6222     | 392.2100  |
| 310P          | bs10              | hmdb51  | 0.6222     | 794.9064  |