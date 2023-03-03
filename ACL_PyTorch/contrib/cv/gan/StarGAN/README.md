# StarGan模型-推理指导


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

StarGAN是 Yunjey Choi 等人于 17年11月 提出的一个模型。该模型可以实现 图像的多域间的迁移（作者在论文中具体应用于人脸属性的转换）。在 starGAN 之前，也有很多 GAN模型 可以用于 image-to-image，比如 pix2pix（训练需要成对的图像输入），UNIT（本质上是coGAN），cycleGAN（单域迁移）和 DiscoGAN。而 starGAN 使用 一个模型 实现 多个域 的迁移，这在其他模型中是没有的，这提高了图像域迁移的可拓展性和鲁棒性。

    


- 参考实现：

  ```
  url=https://github.com/yunjey/stargan
  commit_id=94dd002e93a2863d9b987a937b85925b80f7a19f
  code_path=contrib/cv/gan/stargan
  model_name=stargan
  ```
  
 


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | img    | RGB_FP32 | batchsize x  3 x 128 x 128| NCHW         |
  | Attr    | INT | batchsize x 5| NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 3 x 128 x 128 | RGB_FP32  | NCHW           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17(NPU驱动固件版本为6.0.RC1)  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | Pytorch                                                      | 1.5.0   | -                                                            |
                                               



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>


1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   数据集默认路径为 `./data/celeba.zip` ，使用脚本 `unzip_dataset.sh` 解压数据集。

   ```
   bash unzip_dataset.sh
   ```
   数据目录结构请参考：
   ```
   ├──celeba
    ├──images
    ├──list_attr_celeba.txt
   ```
2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。
   使用脚本 `StarGAN_pre_processing.py` 获得二进制 bin 文件和基准的图片结果。
   ```
   python3 StarGAN_pre_processing.py --mode test  --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                  --model_save_dir './200000-G.pth' --result_dir './result_baseline' \
                  --attr_path './data/celeba/list_attr_celeba.txt' --celeba_image_dir './data/celeba/images'  --batch_size 16
   
   ```
   > 本处以--batch_size 16为例，其他batchsize再此修改
   - 参数说明：

     - --mode: 测试参数。
     - --selected_attrs：图像参数。
     - --model_save_dir：模型保存文件路径。
     - --result_dir：基准图像文件夹。
     - --celeba_image_dir：数据集路径。
     - --attr_path：标签文件路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [200000-G.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/StarGan/PTH/200000-G.pth)

   2. 导出onnx文件。

      1. 使用StarGAN_pth2onnx.py导出onnx文件。

         运行StarGAN_pth2onnx.py脚本。

         ```
         python3 StarGAN_pth2onnx.py --input_file './200000-G.pth' --output_file './StarGAN.onnx'
         ```

         获得StarGAN.onnx文件，受模型前处理影响，不支持bs32,64.



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
         atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs1 --input_format=NCHW \
            --input_shape="real_img:1,3,128,128;attr:1,5" --log=debug --soc_version=$\{chip\_name\}      
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input_format：输入数据的格式。
           -   --input_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc_version：处理器型号。

           运行成功后生成StarGAN_bs1.om模型
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      以batchsize=16 为例子
      ```
      python3  -m ais_bench --model ./StarGAN_bs16.om  --input ./bin/img,./bin/attr --output ./  --outfmt TXT  --batchsize 16 --output_dirname result
      ```

      - 参数说明：

      - --model: OM模型路径。
      - --input: 存放预处理bin文件的目录路径
      - --output: 存放推理结果的目录路径
      - --batchsize：每次输入模型的样本数
      - --outfmt: 推理结果数据的格式
      - --output_dirname: 输出结果子目录
        推理后的输出默认在当前目录result下。


   3. 精度验证。

      调用 StarGAN_post_processing.py 来进行后处理，把输出的 txt 文件转换为输出图像。

      ```
      python3 StarGAN_post_processing.py --folder_path result --batch_size 16
      ```

     详细的结果输出在 `./output_[yourBatchSize]/jpg` 文件夹中，可以和 `result_baseline` 文件夹下的在线推理结果做对比。可以发现各个 batchsize 的离线推理生成的图片与基准基本一致。
    
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model StarGAN_bs1.om --loop 100 --batchsize 1
      ```

      -参数说明：
       + --model: om模型
       + --batchsize: 每次输入模型样本数
       + --loop: 循环次数    



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Batch Size | 310 (FPS/Card) | 310P (FPS/Card)| 
| ---------- | -------------- | ------------- | 
| 1          | *189.753*      | *1025.6*     | 
| 4          | *201.207*      | *1280*     | 
| 8          | *199.913*      | *1281.4*     | 
| 16         | *200.986*      | *1244.8*     |


精度测试：

baseline
![输入图片说明](https://foruda.gitee.com/images/1661163438787898836/屏幕截图.png "屏幕截图.png")

310P：

bs1:

![输入图片说明](https://foruda.gitee.com/images/1662535055936672123/ab08df50_5666861.png "屏幕截图")