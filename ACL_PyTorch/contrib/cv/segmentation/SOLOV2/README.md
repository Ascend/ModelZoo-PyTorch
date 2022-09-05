# {SOLOV2}模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******

  


# 概述

SOLOV2模型是一个box-free的实例分割模型。SOLOV2相对SOLOV1的主要改动有两点，一是通过一个有效的整体实例掩码表示方案来实现，该方案动态地分割图像中的每个实例，而不需要使用边界盒检测。 具体来说，目标掩码的生成（Mask generation）分解为掩码核预测（Mask kernel prediction）和掩码特征学习（Mask feature learning），分别负责生成卷积核和待卷积的特征映射。二是SOLOV2通过我们的新矩阵显著减少了推理开销非最大抑制（NMS）技术。


- 参考实现：

  ```
  url=https://github.com/WXinlong/SOLO
  branch=master
  commit_id=95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
  ```



  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 800 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 大小            | 数据类型 | 数据排布格式 |
  | -------- | --------------- | -------- | ------------ |
  | output1  | 100 x 200 x 304 | FLOAT32  | ND           |
  | output2  | 100             | INT32    | ND           |
  | output3  | 100             | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.8.13  | -                                                            |
| PyTorch                                                      | 1.9.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

**表 2** 依赖列表

| 依赖名称        | 版本     |
| --------------- | -------- |
| onnx            | 1.9.0    |
| onnxruntime     | 1.8.0    |
| onnx-simplifier | 0.3.6    |
| Torch           | 1.9.0    |
| TorchVision     | 0.10.0   |
| Cython          | 0.29.25  |
| Opencv-python   | 4.5.4.60 |
| pycocotools     | 2.0.3    |
| numpy           | 1.22.4   |
| mmcv            | 0.2.16   |
| mmdet           | 1.0.0    |
| Pytest-runner   | 5.3.1    |
| protobuf        | 3.20.0   |
| decorator       | 5.1.1    |
| sympy           | 1.10.1   |

其中mmcv需要用以下方式安装。

```
git clone https://github.com/open-mmlab/mmcv -b v0.2.16
cd mmcv
python setup.py build_ext
python setup.py develop
cd ..
```

其中mmdet需要用以下方式安装。

```
cd SOLO
patch -p1 < ../MMDET.diff
patch -p1 < ../SOLOV2.diff
pip install -r requirements/build.txt # 可以不用执行这句话
pip install -v -e .
cd ..
```

> **说明：**
>
> 请用户根据自己的运行环境自行安装所需依赖。

# 快速上手

#### 获取源码

1. 单击“立即下载”，下载源码包。

2. 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。

   ```
├── benchmark.aarch64            //离线推理工具（适用ARM架构） 
   ├── benchmark.x86_64             //离线推理工具（适用x86架构） 
   ├── pth2onnx.py               //pth文件转onnx脚本
   ├── solov2_postprocess.py  //数据集后处理脚本，同时得出精度数据
   ├── solov2_preprocess.py   //数据集预处理脚本
   ├── get_info.py          //生成推理输入的数据集二进制bin文件与info文件
   ├── README.md  
   ├── requirements.txt
   ├── modelzoo_level.txt
   ├── MMDET.diff        //由于原版的mmdet无法在npu上运行，这个diff文件中包含能让mmdet在npu上运行的修改
   ├── SOLOV2.diff       //SOLOv2模型修改
   ├── SOLOv2_R50_1x.pth  //pth文件
   ├── customize_dtypes.cfg //算子调优文件
   ```

> **说明：**
>
> benchmark离线推理工具使用请参见《[CANN 推理benchmark工具用户指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
>
> 

3. 获取开源代码仓。

   在已下载的源码包根目录下，执行如下命令。

   ```
git clone https://github.com/WXinlong/SOLO.git -b master
   cd SOLO
   git reset --hard 95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
   cd ..
   ```
   
   

## 准备数据集

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型需要coco2017数据集，数据集下载地址https://cocodataset.org/。请将val2017图片及其标注文件放入服务器/root/dataset/coco/文件夹，val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：

   ```
   ├──root
       └──dataset
           └──coco
              └──annotations
              └──val2017
   ```

   

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行“solov2_preprocess.py”脚本，完成预处理。

   ```
   python3 solov2_preprocess.py --image_src_path=/root/dataset/coco/val2017  --bin_file_path=val2017_bin --meta_file_path=val2017_bin_meta --model_input_height=800  --model_input_width=1216
   ```

   

   * --image_src_path：数据集路径

   * --bin_file_path：生成的图片bin文件路径

   * --meta_file_path：生成的图片附加信息路径（临时信息，get_info.py需要用到）

   每个图像对应生成一个二进制bin文件，一个附加信息文件。

3. 生成数据集info文件。

   生成数据集info文件，执行“get_info.py”，会生成两个文件，其中“solo.info”用于benchmark执行，“solov2_meta.info”用于后处理。

   ```
   python3 get_info.py /root/dataset/coco/  SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py  val2017_bin  val2017_bin_meta  solov2.info  solov2_meta.info  1216 800
   ```

   

   * “/root/dataset/coco/”：数据集路径。

   * “SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py”：模型配置文件。

   * “val2017_bin”：预处理后的数据文件的**相对路径**。

   * “val2017_bin_meta”：预处理后的数据文件的**相对路径**。

   * solo.info：生成的数据集文件保存的路径。

   * solo2_meta.info：生成的数据集文件保存的路径。

   * “1216”：图片宽。

   * “800”：图片高。
   
   运行成功后，在当前目录中生成“solo.info”和“solov2_meta.info”。


## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件：“SOLO_R50_1x.pth”，请将其放在与“pth2onnx.py”文件同一目录内。

   2. 安装环境

       由于原版的MMDET无法在NPU环境中运行，我们对MMDET进行了修改，请确保已经利用MMDET.diff文件对其进行修改并安装完必要的环境。

   3. 导出onnx文件。

      1. 使用“SOLO_R50_1x.pth”，导出onnx文件。

         运行“pth2onnx.py”脚本。

         ```
         python3 pth2onnx.py --config SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py --pth_path SOLOv2_R50_1x.pth --out SOLOv2.onnx --shape 800 1216
         ```

         获得“SOLOv2.onnx文件。

      2. 优化onnx模型。

         ```
         python3 -m onnxsim SOLOv2.onnx SOLOv2_sim.onnx
         ```

         获得“SOLOv2_sim.onnx”文件。

         > **须知：**
         >
         > 使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。

   4. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
         ```

         > **说明： **
         >
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《CANN 开发辅助工具指南 (推理)》。
         
      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         ​```
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
         创建customize_dtypes.cfg文件
         ```

         文件内容：  OpType::MatMulV2:InputDtype:float16,OutputDtype:float16

         ```
         atc --framework=5 --model=SOLOv2_sim.onnx --output=solov2  --input_format=NCHW --input_shape="input:1,3,800,1216" --customize_dtypes=customize_dtypes.cfg --precision_mode=force_fp16 --log=error --soc_version=Ascend${chip_name}
         ```

           - 参数说明：
           
         -   --model：为ONNX模型文件。
         - --framework：5代表ONNX模型。
         - --output：输出的OM模型。
         - --input\_format：输入数据的格式。
         - --input\_shape：输入数据的shape。
         - --log：日志级别。
         - --soc\_version：处理器型号。
         - --customize_dtypes：自定义算子的计算精度。
          - --precision_mode：其余算子的精度模式。

           运行成功后生成“solov2.om”模型文件。

2. 开始推理验证。

     a.  使用ais-infer工具进行推理。

     ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

     b.  执行推理。

     执行命令（纯推理场景）：

     ```
     python3 ais_infer.py  --model /home/ysl/SOLOV2/solov2.om --output /home/ysl/SOLOV2/result --outfmt BIN --loop 1
     ```

     参数说明：

     * --model 	需要进行推理的om模型
     * --output 	推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
     * --outfmt 	输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”
     * --loop 	推理次数，可选参数，默认1，profiler为true时，推荐为1

     推理后的输出默认在当前目录result下。
     执行命令（文件夹输入）：

     ```
     python3 ais_infer.py --model "/home/ysl/SOLOV2/solov2.om" --input "/home/ysl/SOLOV2/val2017_bin/" --output "/home/ysl/SOLOV2/result/" --outfmt BIN --loop 1
     ```

     参数说明：

     * --model 	需要进行推理的om模型

     * --input		图片bin文件路径
     * --output 	推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
     * --outfmt 	输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”
     * --loop 	推理次数，可选参数，默认1，profiler为true时，推荐为1	
         推理后的输出默认在当前目录result下。

     c.  精度验证。
     执行solov2_preprocess.py文件中，脚本执行完毕后即可生成精度结果。

     ```
     python3 solov2_postprocess.py  --dataset_path=/root/dataset/coco/   --model_config=SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py  --bin_data_path=./result/2022_09_02-17_04_20/  --meta_info=solov2_meta.info  --net_out_num=3  --model_input_height 800  --model_input_width 1216
     ```

     - --dataset_path：数据集路径。
     - --model_config：模型配置文件。
     - --bin_data_path：执行推理后的数据文件的**相对路径**。
     - --meta_info：生成的数据集文件保存的路径。
     - --net_out_num：输出节点数。
     - --model_input_height：图片高。
     - --model_input_width ：图片宽。


# 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集   | 精度   | 性能        |
| -------- | ---------- | -------- | ------ | ----------- |
| 310P3    | 1          | coco2017 | 34.0％ | 17.3309 fps |

备注：离线模型不支持多batch。