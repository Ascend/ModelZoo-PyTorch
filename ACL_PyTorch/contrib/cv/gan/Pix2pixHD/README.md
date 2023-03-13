# Pix2pixHD模型-推理指导


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

Pix2PixHD作为pix2pix的改进版本，是一个经典的图像生成网络，主要用来产生高分辨率的图像。该网络的突出之处在于：使用多尺度的生成器以及判别器等方式从而生成高分辨率图像；使用了一种非常巧妙的方式，实现了对于同一个输入，产生不同的输出。并且实现了交互式的语义编辑方式，这一点不同于pix2pix中使用dropout保证输出的多样性。这些特点能够让Pix2PixHD生成较pix2pix更高分辨率和含有更多丰富细节信息的图像。

- 论文参考： [Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, Bryan Catanzaro. High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1711.11585v1) 


- 参考实现：

  ```
  url=https://github.com/NVIDIA/pix2pixHD
  branch=master
  commit_id=5a2c87201c5957e2bf51d79b8acddb9cc1920b26
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 36 x 1024 x 2048 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 3 x 1024 x 2048 | NCHW           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/NVIDIA/pix2pixHD
   cd pix2pixHD
   git checkout master
   git reset --hard 5a2c87201c5957e2bf51d79b8acddb9cc1920b26

   patch -p1 < ../pix2pixhd_npu.diff
   cd ..
   ```
   目录结构如下：
   ```
   ├──pix2pixHD                         //开源仓目录
   ├──pix2pixhd_pth2onnx.py
   ├──pix2pixhd_preprocess.py
   ├──pix2pixhd_postprocess.py
   ├──pix2pixhd_npu.diff
   ├──pix2pixhd_gpu.diff
   ├──datasets_deal.py
   ├──LICENCE
   ├──requirements.txt
   ├──README.md
   ├──modelzoo_level.txt
   ```

2. 安装依赖。

   ```用cityscapes数据集的
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   1. 本模型支持[cityscapes 数据集](https://gitee.com/link?target=https%3A%2F%2Fwww.cityscapes-dataset.com%2Fdownloads%2F)。可上传数据集至任意位置下（以"./dataset"下为例），目录结构如下：

      ```
      ├──datasets
            ├──gtFine
                  ├──train
                  ├──val     //支持的数据集
                  ├──test
      ```
   2. 原仓已经处理的数据集的验证，不需要下载。
      ```
      ├──./pix2pixHD
           ├──datasets
               ├──cityscapes
                     ├──test_inst
                     ├──test_label
      ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   依次执行脚本，完成cityscapes数据集的处理。

   1. 数据集第一步处理（使用原仓已经处理的数据集可不用执行）。
      ```
      rm -rf ./pix2pixHD/datasets/cityscapes 
      python3 datasets_deal.py ./pix2pixHD/datasets/cityscapes/test_inst ./pix2pixHD/datasets/cityscapes/test_label ./datasets/gtFine/val
      ```
      - 参数说明：
         - 第一个参数：对数据集的第一步处理,test_ins文件夹名称不支持修改（与test_label放于同一文件夹下）。
         - 第二个参数：对数据集的第一步处理,test_label文件夹名称不支持修改（与test_ins放于同一文件夹下）。
         - 第三个参数：原数据集的路径。
   2. 数据集预处理。  
      ```
      python3 pix2pixhd_preprocess.py ./pix2pixHD/datasets/cityscapes ./prep_datasets
      ```
      - 参数说明：
         - 第一个参数：数据集文件路径。
         - 第二个参数：处理后的文件路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载[权重文件latest_net_G.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Pix2pixHD/PTH/latest_net_G.pth),放置"./pix2pixHD/checkpoints/label2city_1024p/"路径下。
       

   2. 导出onnx文件。

      1. 使用pix2pixhd_pth2onnx.py导出onnx文件。

         运行pix2pixhd_pth2onnx.py脚本。

         ```
         python3 pix2pixhd_pth2onnx.py --load_pretrain ./pix2pixHD/checkpoints/label2city_1024p/ --output_file pix2pixhd.onnx
         ```
         - 参数说明：
            - 第一个参数：权重文件路径。
            - 第二个参数：生成的onnx文件路径名称。
         在当前路径下获得pix2pixhd.onnx文件。


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
         atc --framework=5 --model=./pix2pixhd.onnx --input_format=NCHW --input_shape="input_concat:1,36,1024,2048" --output=pix2pixhd_bs1 --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成pix2pixhd_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      mkdir result
      python3 -m ais_bench --model=./pix2pixhd_bs1.om --input=./prep_datasets/ --output=./result/ --output_dirname=bs1 --outfmt=BIN --batchsize=1  --device 0 
      ```
      - 参数说明：
         - --model：模型类型。
         - --input：om模型推理输入文件路径。
         - --output：om模型推理输出文件路径。
         - --output_dirname：
         - --outfmt：输出格式
         - --batchsize：批大小。
         - --device：NPU设备编号。

        推理后的结果在"./result/bs1/"路径下。



   3. 精度验证。

      1. 调用脚本生成om模型推理输出文件的复原图片。

         ```
         python3 pix2pixhd_postprocess.py ./result/bs1 ./result/bs1generated
         ```
         - 参数说明：
            - 第一个参数：om模型推理输出文件路径。
            - 第二个参数：om模型推理输出文件还原图片的结果路径。

         后处理om模型推理输出文件的结果在"./result/bs1generated/"路径下。
      2. 通过在线推理生成原仓的推理结果复原图片。
         ```
         cd pix2pixHD
         bash ./scripts/test_1024p.sh
         ```

         在线推理结果在"./results/label2city_1024p/test_latest/images/"路径下。

      通过观察在线推理与om模型推理输出文件的复原图片对比，验证精度。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size} --device 0
      ```
      - 参数说明：
         - --model：om模型的路径。
         - --loop：推理的循环次数。
         - --batch_size：批大小。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | :----------------: | ---------- | ---------- | --------------- |
|  Ascend310P   |  1   |   cityscapes    |    通过观察图片精度达标    |       5.106     |
|  Ascend310P   |  4   |   cityscapes    |    通过观察图片精度达标    |       4.897     |
|  Ascend310P   |  8   |   cityscapes    |    通过观察图片精度达标    |       4.863     |

注：因内存原因只测试到batch_size8。