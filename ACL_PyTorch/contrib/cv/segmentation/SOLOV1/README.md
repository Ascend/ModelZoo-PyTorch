# SOLOV1模型-推理指导


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

SOLOV1模型是一个box-free的实例分割模型，其引入“实例类别”的概念来区分图像中的对象实例，即量化的中心位置和对象大小，这使得可以利用位置来分割对象。与其他端到端的实例分割模型相比，其达到了竞争性的准确性。


- 参考实现：

  ```
  url=https://github.com/WXinlong/SOLO
  branch=master 
  commit_id= 95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
  model_name=SOLOV1
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
  | input    | RGB_FP32 | batchsize x 3 x 800 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 100x200x304 | FLOAT32  | ND           |
  | output2  | 100         | INT32    | ND           |
  | output3  | 100         | FLOAT32  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.13   | -                                                            |
| PyTorch                                                      | 1.9.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>


1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```

2. 获取，修改与安装开源模型代码  
   安装mmcv
   ```
   git clone https://github.com/open-mmlab/mmcv -b v0.2.16
   cd mmcv
   python setup.py build_ext
   python setup.py develop
   cd ..
   ```
   获取SOLOv1代码
   ```
   git clone https://github.com/WXinlong/SOLO.git -b master
   cd SOLO
   git reset --hard 95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
   patch -p1 < ../MMDET.diff
   patch -p1 < ../SOLOV1.diff
   pip install -r requirements/build.txt
   pip install -v -e .
   cd ..
   ```


## 准备数据集<a name="section183221994411"></a>

1.获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   数据集的获取请参考[原始开源代码仓](https://github.com/WXinlong/SOLO)的方式获取。请将val2017图片及其标注文件放入服务器/root/dataset/coco/文件夹，val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：
   ```
   root
   ├── dataset
   │   ├── coco
   │   │   ├── annotations
   │   │   ├── val2017

   ```

2.数据预处理。
   将原始数据集转换为模型输入的二进制数据。执行“solov1_preprocess.py”脚本。
   ```
   python3 solov1_preprocess.py --image_src_path=/root/dataset/coco/val2017  --bin_file_path=val2017_bin --meta_file_path=val2017_bin_meta --model_input_height=800  --model_input_width=1216
   
   ```
   - --image_src_path：数据集路径
   - --bin_file_path：生成的图片bin文件路径
   - --meta_file_path：生成的图片附加信息路径（临时信息，get_info.py需要用到）
   每个图像对应生成一个二进制bin文件，一个附加信息文件。
   

3.生成数据集info文件。
   生成数据集info文件，执行“get_info.py”，会生成两个文件，其中“solo.info”用于benchmark执行，“solo_meta.info”用于后处理。
   ```
   python3 get_info.py /root/dataset/coco/  SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py  val2017_bin  val2017_bin_meta  solo.info  solo_meta.info  1216 800

   ```
   - --“/root/dataset/coco/”：数据集路径。

   - --“SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py”：模型配置文件。

   - --“val2017_bin”：预处理后的数据文件的相对路径。

   - --“val2017_bin_meta”：预处理后的数据文件的相对路径。

   - --“1216”：图片宽。

   - --“800”：图片高。

   运行成功后，在当前目录中生成“solo.info”和“solo_meta.info”。


## 模型推理<a name="section741711594517"></a>

1.模型转换。
   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1.获取权重文件。
       从源码包中获取权重文件：“SOLO_R50_1x.pth”，请将其放在与“pth2onnx.py”文件同一目录内。从（https://github.com/WXinlong/SOLO）下载

    2.导出onnx文件。
       1.使用“SOLO_R50_1x.pth”导出onnx文件。

         运行“pth2onnx.py”脚本。

         ```
         python3 pth2onnx.py --config SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py --pth_path SOLO_R50_1x.pth --out SOLOv1.onnx --shape 800 1216

         ```

         获得“SOLOv1.onnx”文件。

       2.优化ONNX模型。

         ```
         python3 -m onnxsim SOLOv1.onnx SOLOv1_sim.onnx

         ```

         获得SOLOv1_sim.onnx文件。

    3.使用ATC工具将ONNX模型转OM模型。

       1.配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh

         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

       2.执行命令查看芯片名称（$\{chip\_name\}）。

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

       3.执行ATC命令。

         ```
         atc --framework=5 --model=SOLOv1_sim.onnx --output=solo  --input_format=NCHW --input_shape="input:1,3,800,1216" --log=error --soc_version=Ascend${chip_name}

         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input_format：输入数据的格式。
           -   --input_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc_version：处理器型号。
           -   --insert_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后生成“solo.om”模型文件。



 2.开始推理验证。

    a. 使用ais-infer工具进行推理。
       ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


    b.  执行推理。

       python3 ais_infer.py --model "/home/cc/SOLOV1/soloc.om" --input "/home/cc/SOLOV1/val2017_bin/" --output "/home/cc/SOLOV1/result/" --outfmt BIN --device 0 --batchsize 1 --loop 1

       -   参数说明：
       -   --model：om文件路径。
       -   --input:输入路径
       -   --output：输出路径。
       推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

    c.  精度验证。

       调用脚本与数据集val2017标签比对

       ```
       python3 solov1_postprocess.py  --dataset_path=/root/dataset/coco/   --model_config=SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py  --bin_data_path=./result/ 2022_09_03-10_09_16/  --meta_info=solo_meta.info  --net_out_num=3  --model_input_height 800  --model_input_width 1216
       ```
       - --result/2022_09_03-10_09_16/：为生成推理结果所在路径  
       - --val2017：为标签数据
    


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|  310P3    |   bs1               |    val2017        |    32.1%        |     10.3064   |
