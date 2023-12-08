# SOLOV2模型-基于推理引擎PyTorch框架插件的部署及推理指导

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

SOLOV2模型是一个box-free的实例分割模型。SOLOV2相对SOLOV1的主要改动有两点，一是通过一个有效的整体实例掩码表示方案来实现，该方案动态地分割图像中的每个实例，而不需要使用边界盒检测。 具体来说，目标掩码的生成（Mask generation）分解为掩码核预测（Mask kernel prediction）和掩码特征学习（Mask feature learning），分别负责生成卷积核和待卷积的特征映射。二是SOLOV2通过我们的新矩阵显著减少了推理开销非最大抑制（NMS）技术。


- 参考实现：

  ```
  url=https://github.com/WXinlong/SOLO
  branch=master
  commit_id=95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
  ```

## 输入输出数据<a name="section540883920406"></a>

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


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| 固件与驱动                                                      | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 7.0.RC1.alpha003 | -                                                                                                     |
| Python                                                          | 3.9.11  | -                                                                                                     |
| PyTorch                                                         | 2.0.1   | -                                                                                                     |
| torch_aie                                                       | 6.3.rc2  | -                                                                                           |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码

1. 获取开源代码仓。

   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/WXinlong/SOLO.git -b master
   cd SOLO
   git reset --hard 95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
   cd ..
   ```

2. 安装依赖。

   ```
   apt-get install libjpeg-dev zlib1g-dev libgl1-mesa-glx
   pip install -r requirements.txt
   ```

   其中mmcv安装建议参考[官方安装指导说明](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html#pip)
   ```
   # Linux cpu torch2.0.x mmcv2.1.0 (Linux环境安装指令)
   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
   ```

   其中mmdet需要用以下方式安装。

   ```
   cd SOLO
   patch -p1 < ../MMDET.diff
   patch -p1 < ../SOLOV2.diff
   pip install -v -e .
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型需要coco2017数据集，数据集下载地址https://cocodataset.org/

   请将val2017图片及其标注文件放入服务器/data/datasets/coco/文件夹，val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：

   ```
   ├──root
       └──dataset
           └──coco
              └──annotations
              └──val2017
   ```


2. 数据预处理。

   ```
   python3 solov2_preprocess.py                    \
      --image_src_path=/data/datasets/coco/val2017 \
      --bin_file_path=val2017_bin                  \
      --meta_file_path=val2017_bin_meta            \
      --model_input_height=800                     \
      --model_input_width=1216
   ```

   - --image_src_path：数据集路径
   - --bin_file_path：生成的图片bin文件路径
   - --meta_file_path：生成的图片附加信息路径（临时信息，get_info.py需要用到）

   每个图像对应生成一个二进制bin文件，一个附加信息文件。

3. 生成数据集info文件。

   执行“get_info.py”，会生成“solov2_meta.info”用于后处理。

   ```
   python3 solov2_get_info.py /data/datasets/coco/  SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py  val2017_bin  val2017_bin_meta  solov2.info  solov2_meta.info  1216 800
   ```

   * “/data/datasets/coco/”：数据集路径。

   * “SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py”：模型配置文件。

   * “val2017_bin”：预处理后的数据文件的**相对路径**。

   * “val2017_bin_meta”：预处理后的数据文件的**相对路径**。

   * solo.info：生成的数据集文件保存的路径。

   * solo2_meta.info：生成的数据集文件保存的路径。

   * “1216”：图片宽。

   * “800”：图片高。

   运行成功后，在当前目录中生成“solov2_meta.info”。


## 模型推理<a name="section741711594517"></a>

### 1. 模型转换

   使用PyTorch将模型权重文件.pth转换为torchscript文件

   1. 获取权重文件。

      权重文件：[SOLOv2_R50_1x.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/SOLOV2/PTH/SOLOv2_R50_1x.pth)，请将其放在与“solov2_pth2torchscript.py”文件同一目   

   2. 导出torchscript文件

      ```shell
      python3 solov2_pth2torchscript.py                         \
         --config SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py \
         --pth-path SOLOv2_R50_1x.pth                           \
         --shape 800 1216
      ```

      获得solov2.torchscript.pt文件。

      + 参数说明
         + `--config`：模型配置文件路径
         + `--pth-path`：PTH权重文件路径
         + `--shape`：模型输入shape

   3. 配置环境变量。

      ```shell
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
      ```

      > **说明：**
      >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   4. 执行命令查看芯片名称（$\{chip\_name\}）。

      ```shell
      npu-smi info
      #该设备芯片名为Ascend310P3 （在下一步中赋值给soc_version环境变量）
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

   5. 对原生ts文件执行torch_aie编译，导出NPU支持的ts文件

      ```shell
      soc_version="Ascend310P3" # User-defined
      python solov2_export_torch_aie_ts.py           \
         --torch-script-path ./solov2_torchscript.pt \
         --batch-size 1                              \
         --save-path ./                              \
         --soc-version ${soc_version}
      ```

      + 参数说明
         + `--torch-script-path`：原生ts文件路径
         + `--batch-size`：用户自定义的batch size
         + `--save-path`：AIE编译后的ts文件保存路径
         + `--soc-version`：NPU型号

      运行成功后生成solov2_torchscriptb1_torch_aie.pt模型文件。

### 2. 执行推理并验证精度与性能

   1.  执行推理

   推理完成后将输出模型推理性能结果

   ```shell
   python solov2_inference.py                               \
      --aie-module-path ./solov2_torchscriptb1_torch_aie.pt \
      --batch-size 1                                        \
      --processed-dataset-path ./val2017_bin/               \
      --output-save-path ./result_aie/                      \
      --device-id 0
   ```

   + 参数说明：
      + --aie-module-path: AIE编译后模型的路径
      + --batch-size: 模型输入的BatchSize
      + --processed-dataset-path：经预处理COCO数据集的路径
      + --output-save-path：推理结果保存路径
      + --device-id: Ascend NPU ID(可通过npu-smi info查看)

   2.  数据后处理

   处理完成后将输出模型推理精度结果

   ```shell
   python solov2_postprocess.py                                    \
      --dataset_path /data/datasets/coco/                          \
      --model_config SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py \
      --bin_data_path result_aie                                   \
      --meta_info solov2_meta.info                                 \
      --net_out_num 3                                              \
      --model_input_height 800                                     \
      --model_input_width 1216
   ```
 
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

基于推理引擎完成推理计算，精度与性能可参考下列数据：

| Soc version | Batch Size | Dataset | Accuracy | Performance |
| ----------  | ---------- | ---------- | ---------- | ---------- |
| Ascend310P3  | 1  | coco2017 | Average Precision(IoU=0.50:0.95): 0.340 | 26.19 fps |


# FAQ
1. 若遇到类似报错：ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block

   解决方法：
   export LD_PRELOAD=$LD_PRELOAD:{报错信息中的路径}