# U-2-Net模型-推理指导


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

U-2-Net是基于UNet提出的一种新的网络结构，网络基于encode-decode，结合了FPN/UNet提出了RSU结构，在物体前背景分割任务上取得了良好结果，且具备了较好的实时性。

- 参考论文：[Qin X , Zhang Z , Huang C , et al. U2-Net: Going deeper with nested U-structure for salient object detection[J]. Pattern Recognition, 2020, 106:107404.](https://sci-hub.se/10.1016/j.patcog.2020.107404)

- 参考实现：

   ```
   url=https://github.com/xuebinqin/U-2-Net
   branch=master
   commit_id=a179b4bfd80f84dea2c76888c0deba9337799b60
   ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 320 x 320 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 3 x 320 x 320 | NCHW          |



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
   mkdir workspace && cd workspace
   git clone https://github.com/xuebinqin/U-2-Net.git -b master
   cd U-2-Net
   git reset --hard a179b4bfd80f84dea2c76888c0deba9337799b60
   cd ../../
   ```
   
   目录结构如下：
   ```
   ├──workspace
      ├──U-2-Net
   ├──pth2onnx.py
   ├──fix_onnx.py                              
   ├──preprocess.py
   ├──postprocess.py
   ├──evaluate.py
   ├──README.md
   ├──LICENSE
   ├──requirements.txt
   ├──modelzoo_level.txt
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[ECSSD 数据集](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)。用户可自行下载至任意数据集，以"./datasets"为例。目录结构如下：

   ```
   ├──./datasets
      ├──ECSSD
         ├──images
         ├──ground_truth_mask
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```
   python3 preprocess.py --image_dir ./datasets/ECSSD/images --save_dir ./test_data_ECSSD
   ```
   - 参数说明：
      - --image_dir：数据集路径。
      - --save_dir：预处理结果保存的路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载[权重文件 u2net.pth](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing)或[百度云 提取码:pf9k](https://pan.baidu.com/s/1WjwyEwDiaUjBbx_QxcXBwQ)放置于"./workspace/U-2-Net/saved_models/u2net/"路径下。

   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         mkdir models
         python3 pth2onnx.py --model_dir=./workspace/U-2-Net/saved_models/u2net/u2net.pth --out_path=./models/u2net.onnx
         ```

         获得./models/u2net.onnx文件。

      2. 优化ONNX文件(需要用到[auto-optimizer工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)，请自行下载安装)。

         ```
         python3 -m onnxsim models/u2net.onnx models/u2net_sim_bs1.onnx --input-shape "image:1,3,320,320"
         python3 fix_onnx.py models/u2net_sim_bs1.onnx models/u2net_sim_bs1_fixv2.onnx
         ```

         获得./models/u2net_sim_bs1_fixv2.onnx文件。

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
         atc --framework=5 --model=./models/u2net_sim_bs1_fixv2.onnx --output=./models/u2net_sim_bs1_fixv2 --input_format=NCHW --input_shape="image:1,3,320,320" --out_nodes='Sigmoid_1048:0' --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input\_format：输入数据的格式。
            - --input\_shape：输入数据的shape。
            - --log：日志级别。
            - --soc\_version：处理器型号。
            - --out\_nodes: 指定输出节点。

           运行成功后生成./models/u2net_sim_bs1_fixv2.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```
      mkdir result  
      source /usr/local/Ascend/ascend-toolkit/set_env.sh  
      python3 -m ais_bench --model=./models/u2net_sim_bs1_fixv2.om --input=./test_data_ECSSD/ --output=./result/ --output_dirname=bs1 --outfmt=BIN --batchsize=1  --device 0
      ```

      - 参数说明：
         - --model：模型类型。
         - --input：om模型推理输入文件路径。
         - --output：om模型推理输出文件路径。
         - --output_dirname：om模型推理输出文件路径的子文件夹。
         - --outfmt：输出格式
         - --batchsize：批大小。
         - --device：NPU设备编号。

        推理后的输出默认在当前目录./result/bs1下。


   3. 精度验证。

      1. 对om的推理的结果进行复原。
         ```
         python3 postprocess.py --image_dir ./datasets/ECSSD/images --save_dir ./test_vis_ECSSD_bs1 --out_dir ./result/bs1
         ```

         - 参数说明：
            - --image_dir：数据集图片路径。
            - --save_dir：om模型推理结果生成的复原图片路径。
            - --out_dir：om模型的推理结果文件。

      2. 精度验证。
         ```
         python3 evaluate.py --res_dir ./test_vis_ECSSD_bs1 --gt_dir ./datasets/ECSSD/ground_truth_mask
         ```

         - 参数说明：
            - --res_dir：om的推理的结果复原图片的路径。
            - --gt_dir：数据集真实值路径。

      推理om模型的精度数据会打印在终端屏幕上。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m  ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size} --device=0
      ```

      - 参数说明：
         - --model：om模型路径。
         - --loop：循环次数。
         - --batchsize：批大小
         - --device：NPU设备编号。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| --------- | :---------: | ---------- | ---------- | --------------- |
|  Ascend310P  |  1  |  ECSSD  |  maxF:94.8% MAE:0.033  |   240.978   |
|  Ascend310P  |  4  |  ECSSD  |  maxF:94.8% MAE:0.033  |   208.310   |
|  Ascend310P  |  8  |  ECSSD  |  maxF:94.8% MAE:0.033  |   202.463   |
|  Ascend310P  |  16  |  ECSSD  |  maxF:94.8% MAE:0.033  |   198.603   |
|  Ascend310P  |  32  |  ECSSD  |  maxF:94.8% MAE:0.033  |   197.363   |
|  Ascend310P  |  64  |  ECSSD  |  maxF:94.8% MAE:0.033  |   199.131   |