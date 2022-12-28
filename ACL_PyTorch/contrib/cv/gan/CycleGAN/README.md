# CycleGAN模型-推理指导


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

CycleGAN是基于对抗生成的图像风格转换卷积神经网络，该网络具有两个生成器，这两个生成器可以互相转换图像风格。该网络的训练是一种无监督的，少样本也可以取得很好效果的网络。

- 论文参考： [Isola P, Zhu J Y, Zhou T, et al. Image-to-image translation with conditional adversarial networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1125-1134.](http://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html) 


- 参考实现：

   ```
   url=https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
   branch=master
   commit_id=9bcef69d5b39385d18afad3d5a839a02ae0b43e7
   model_name=CycleGAN
   ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32  | batchsize x 3 x 256 x 256 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | output  | FLOAT32  | batchsize x 3 x 256x 256 | NCHW        |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

   | 配套                                                         | 版本    | 环境准备指导                                                 |
   | :------------------------------------------------------------: | :-------: | :------------------------------------------------------------: |
   | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
   | CANN                                                         | 5.1.RC2 | -                                                            |
   | Python                                                       | 3.7.5   |  \                                                            |
   | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix -b master
   cd ctpn.pytorch
   git reset --hard 9bcef69d5b39385d18afad3d5a839a02ae0b43e7 

   patch -p1 < ../CycleGAN.patch
   cp ./models/networks.py ../
   cd ..
   ```

   源码目录结构如下：

   ```
   ├──pytorch-CycleGAN-and-pix2pix           //开源仓目录
   ├──CycleGAN_preprocess.py
   ├──CycleGAN_postprocess.py
   ├──CycleGAN_pth2onnx.py
   ├──parse.py
   ├──networks.py
   ├──CycleGAN.patch
   ├──LICENCE
   ├──requirements.txt
   ├──README.md
   ├──modelzoo_level.txt
   ```

   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持[maps 数据集](https://gitee.com/link?target=http%3A%2F%2Fefrosgans.eecs.berkeley.edu%2Fcyclegan%2Fdatasets%2Fmaps.zip)。用户可自行获取maps数据集上传到服务器，可放置于任意路径下，以"./datasets"目录为例。

   ```
   ├──maps
      ├── test
      ├── testA
      ├── testB
      ├── train
      ├── trainA
      ├── trainB
      ├── val
      ├── valA
      ├── valB
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行CycleGAN_preprocess.py脚本，完成预处理。

   ```python
   python3 CycleGAN_preprocess.py --src_path_testA=./datasets/maps/testA/   --save_pathTestA_dst=./datasetsDst/maps/testA/  --src_path_testB=./datasets/maps/testB/ --save_pathTestB_dst=./datasetsDst/maps/testB/
   ```
   - 参数说明：
      - --src_path_testA ：航拍数据转卫星地图的测试集路径。
      - --src_path_testB：卫星地图转航拍的测试集路径。
      - --save_pathTestA_dst：航拍数据转卫星地图的测试集处理后的路径。
      - --save_pathTestB_dst：卫星地图转航拍的测试集处理后的路径。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用Torch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      - [官方权重文件](http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/)。
      - [官方CycleGAN pth权重文件](https://gitee.com/link?target=https%3A%2F%2Fpan.baidu.com%2Fs%2F1YqHkce2wUw-W8_VY9dYD_w)，提取密码为：1234。
      将权重文件可放置于任意位置，以"./"为例


   2. 导出onnx文件。

      1. 使用CycleGAN_pth2onnx.py导导出onnx文件。

         运行CycleGAN_pth2onnx.py导脚本。

         ```python
         python3 CycleGAN_pth2onnx.py --model_ga_path=./latest_net_G_A.pth --model_gb_path=./latest_net_G_B.pth --onnx_path=./   --model_ga_onnx_name=model_Ga.onnx    --model_gb_onnx_name=model_Gb.onnx
         ```

         - 参数说明
            - --model_ga_path：GA权重文件路径。
            - --model_gb_path：GB权重文件路径。
            - --onnx_path：生成的onnx文件路径。
            - --model_ga_onnx_name：GA权重文件生成的模型名。
            - --model_gb_onnx_name：GB权重文件生成的模型名。

         获得model_Ga.onnx和model_Gb.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称。

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
         atc --framework=5 --model=./model_Ga.onnx --output=CycleGAN_Ga_bs1 --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_156:0" --log=error --soc_version=Ascend${chip_name}

         atc --framework=5 --model=./model_Gb.onnx --output=CycleGAN_Gb_bs1 --input_format=NCHW --input_shape="img_maps_sat:1,3,256,256" --out_nodes="Tanh_156:0" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input\_format：输入数据的格式。
            -   --input\_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc\_version：处理器型号。
            -   --out\_nodes：指定输出节点。


         运行成功后生成CycleGAN_Ga_bs1.om和CycleGAN_Gb_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      mkdir resultAbs1 resultBbs1
	  source /usr/local/Ascend/ascend-toolkit/set_env.sh
      python3 -m ais_bench --model=./CycleGAN_Ga_bs1.om --input=./datasetsDst/maps/testA/ --output=./resultAbs1/ --outfmt=BIN --batchsize=1

      python3 -m ais_bench --model=./CycleGAN_Gb_bs1.om --input=./datasetsDst/maps/testB/ --output=./resultBbs1/ --outfmt=BIN --batchsize=1
      ```
      - 参数说明：
         -  --model：om文件路径。
         -  --input：输入的bin文件路径。
         -  --output：推理结果文件路径。
         -  --outfmt：输出结果格式。
         -  --device：NPU设备编号。
         -  --res_dir：得到的结果文件夹。

      推理后的输出在推理结果文件路径下的日期+时间的子文件夹(./resultAbs1/2022_10_28-09_21_37/和./resultBbs1/2022_10_28-09_21_47/)。
   


   3. 精度验证。

      调用脚本与原图片处理后文件比对，可以获得Accuracy数据，结果在标准输出中打印。
    
      ```python
      python3 CycleGAN_postprocess.py --dataroot=./datasets/maps/testA/ --npu_bin_file=./resultAbs1/2022_10_28-09_21_37/ --onnx_path=./ --om_save --onnx_save

      python3 CycleGAN_postprocess.py --dataroot=./datasets/maps/testB/ --npu_bin_file=./resultBbs1/2022_10_28-09_21_47/ --onnx_path=./ --om_save --onnx_save
      ```

      -  参数说明：     
         - --dataroot：前处理后的路径。
         - --npu_bin_file：推理的om模型的结果路径。
         - --onnx_path：以npu_bin_file路径为参考基础的路径。
         - --om_save：存在，则可在./resultBbs1/2022_10_28-09_21_47/./om/下查看om推理生成的图片结果。
         - --onnx_save：存在，则可在./resultBbs1/2022_10_28-09_21_47/./onnx/下查看onnx推理生成的图片结果。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。
- CycleGAN_Ga

   | 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
   | :---------: | :----------------: | :----------: | :----------: | :---------------: |
   |    Ascend310P       |        1          |     maps       |     1.0       |        228.15         |
   |    Ascend310P       |        4          |     maps       |     1.0       |        219.26         |
   |    Ascend310P       |        8          |     maps       |     1.0       |        213.43         |
   |    Ascend310P       |        16          |     maps       |     1.0       |        218.19         |
   |    Ascend310P       |        32          |     maps       |     1.0       |        226.22         |
   |    Ascend310P       |        64          |     maps       |     1.0       |        231.97         |

- CycleGAN_Gb

   | 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
   | :---------: | :----------------: | :----------: | :----------: | :---------------: |
   |    Ascend310P       |    1    |     maps     |     0.9990765      |   228.05         |
   |    Ascend310P       |    4    |     maps     |     0.9990766      |   218.01         |
   |    Ascend310P       |    8    |     maps     |     0.999077       |   206.62         |
   |    Ascend310P       |    16    |     maps    |     0.999077       |   218.42         |
   |    Ascend310P       |    32    |     maps    |     0.999077       |   226.60         |
   |    Ascend310P       |    64    |     maps    |     0.999077       |   232.37         |