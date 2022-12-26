#  FaceBoxes模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)
  - [输入输出数据](#section540883920406)
- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)
- [快速上手](#ZH-CN_TOPIC_0000001126281700)
  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

FaceBoxes的新型人脸检测器，它在速度和准确性方面都有卓越的性能。具体来说，我们的方法拥有一个轻量级但功能强大的网络结构，它由快速消化卷积层(RDCL)和多尺度卷积层(MSCL)组成。RDCL的设计目的是让facebox在CPU上实现实时速度


- 参考论文：[FaceBoxes: A CPU Real-time Face Detector with High Accuracy
  Shifeng Zhang, Xiangyu Zhu, Zhen Lei, Hailin Shi, Xiaobo Wang, Stan Z. Li)](https://arxiv.org/abs/1708.05234)

- 参考实现：

  ```
  url=https://github.com/zisianw/FaceBoxes.PyTorch.git
  branch=master
  commit_id=9bc5811fe8c409a50c9f23c6a770674d609a2c3a
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 大小                      | 数据类型 | 数据排布格式 |
  | -------- | ------------------------- | -------- | ------------ |
  | input    | batchsize x 3 x 224 x 224 | RGB_FP32 | NCHW         |


- 输出数据

  | 输出数据 | 大小             | 数据类型 | 数据排布格式 |
  | -------- | ---------------- | -------- | ------------ |
  | output1  | batchsize x 1024 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套       | 版本                               | 环境准备指导                                                 |
  | ---------- | ---------------------------------- | ------------------------------------------------------------ |
  | 固件与驱动 | 1.0.16（NPU驱动固件版本为5.1.RC2） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | - 
  | Python                                                        | 3.7.5| - 

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   源码目录结构：

    ``` 
    ├── faceboxes_pth2onnx.py             //用于转换pth文件到onnx文件 
    ├── faceboxes_pth_preprocess.py       //数据集预处理脚本
    ├── faceboxes_pth_preprocess.py       //数据集后处理脚本 
    ├── FDDB_Evaluation                   //精度评估文件夹
    ├── aipp.config                       //aipp配置文件
	├── dlt_cuda.patch                    //patch文件
    ```

2. 获取开源代码仓。
   在源码目录下，执行如下命令。

   ```
   git clone https://github.com/zisianw/FaceBoxes.PyTorch.git
   mv faceboxes_pth2onnx.py faceboxes_pth_postprocess.py faceboxes_pth_preprocess.py FDDB_Evaluation/ dlt_cuda.patch aipp.config FaceBoxes.PyTorch/ 
   ```

3. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

4. 切换到主目录。

   ```
   cd FaceBoxes.PyTorch/
   git reset --hard 9bc5811fe8c409a50c9f23c6a770674d609a2c3a
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用[FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) 2845张验证集进行测试，图片与标签分别存放在datasets/FDDB/images（请按下图中的目录放置）与datasets/FDDB/img_list.txt，将ground_truth文件夹放在FDDB_Evaluation目录下。
   数据目录结构请参考：

    ```
    ├── FDDB
       ├──img_list.txt
       ├──images           
           ├── 2002
           ├── 2003   
    ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。
   执行faceboxes_pth_preprocess.py脚本，完成预处理。

   ```
   python faceboxes_pth_preprocess.py --dataset datasets/FDDB --save-folder prep
   ```

   - 参数说明：
       - --dataset：原始数据验证集所在路径。
       - --save-folder：bin文件保存路径。

   运行成功后，在当前目录下生成 prep 二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件 [FaceBoxesProd.pth](https://drive.google.com/file/d/1tRVwOlu0QtjvADQ2H7vqrRwsWEmaqioI)。

   2. 导出onnx文件。

      使用 FaceBoxesProd.pth 导出onnx文件。
      运行 faceboxes_pth2onnx.py 脚本。

      ```
      python faceboxes_pth2onnx.py  --trained_model FaceBoxesProd.pth --save_folder faceboxes-b0.onnx 
      ```

      - 参数说明：
        - --trained_model：pth权重文件所在路径。
        - --save-folder：onnx文件保存路径。

      获得 faceboxes-b0.onnx 文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称(${chip_name})。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 17.6         57                0    / 0              |
         | 0       0         | 0000:3B:00.0    | 0            936 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         通过 [netron](https://gitee.com/link?target=https%3A%2F%2Fnetron.app%2F) 查看onnx的输出节点名称，对应的进行更改--out_nodes里的参数

         ```
         atc --framework=5 --model=faceboxes-b0.onnx --output=faceboxes-b0_bs1 --input_format=NCHW --input_shape="image:1,3,1024,1024" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Reshape_134:0;Softmax_141:0" --enable_small_channel=1 --insert_op_conf=aipp.config
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --enable_small_channel：是否使能small channel的优化，使能后在channel<=4的首层卷积会有性能收益。
           -   --insert_op_conf=aipp_resnet34.config: AIPP插入节点，通过config文件配置算子信息。

        运行成功后生成 faceboxes-b0_bs1.om 模型文件。


2. 开始推理验证。

   1. 安装ais_bench推理工具。请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。。  

   2. 创建推理结果保存的文件夹。

      ```
      mkdir result
      ```

   3. 执行推理。

      ```
      python -m ais_bench --model faceboxes-b0_bs1.om --input prep/ --output result --output_dir dumpout_bs1 --batchsize 1
      ```
      -   参数说明：

           -   --model ：输入的om文件。
           -   --input：输入的bin数据文件。
           -   --device：NPU设备编号。
           -   --output: 模型推理结果。
           -   --batchsize : 批大小。

      推理结果保存在result/dumpout_bs1下面，并且也会输出性能数据。
	  > 说明： 执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见--help命令。


   4. 精度验证。

      1. 在当前目录下，执行以下命令。

         ```
         dos2unix dlt_cuda.patch
         git apply dlt_cuda.patch
         ./make.sh
         ```

      2. 运行后处理脚本faceboxes_pth_postprocess.py。

         ```
         python faceboxes_pth_postprocess.py --save_folder FDDB_Evaluation/ --prep_info prep/ --prep_folder result/dumpout_bs1/
         ```

         - 参数说明：
           - --save_folder：推理结果处理之后的文件保存路径。
           - --prep_info：前处理数据（.bin）文件保存路径。
           - --prep_folder：模型推理结果保存路径。

         运行成功后在FDDB_Evaluation路径下生成FDDB_dets.txt文件。

      3. 执行evaluate.py脚本获得精度数据。

         ```
         cd FDDB_Evaluation
         python setup.py install
         python evaluate.py -g ./ground_truth
         ```

         - 参数说明：
           - --g：ground_truth文件夹的路径。

         执行成功后输出模型精度数据并保存在results.txt文件中。

   5. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

         ```
         python -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
         ```
		 - 参数说明：
           - --model：om模型的路径。
           - --loop：推理循环的次数。
           - --batchsize：推理的batchsize。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
 
  调用ACL接口推理计算，性能参考下列数据。

   | 芯片型号 | Batch Size | 数据集 | 精度  | 性能      |
   | -------- | ---------- | ------ | ----- | --------- |
   | 310P3    | 1          | FDDB   | 0.948 | 2332.9056 |
   | 310P3    | 4          | FDDB   | 0.948 | 1721.6148 |
   | 310P3    | 8          | FDDB   | 0.948 | 1807.0112 |
   | 310P3    | 16         | FDDB   | 0.948 | 1707.1128 |
   | 310P3    | 32         | FDDB   | 0.948 | 1662.0441 |
