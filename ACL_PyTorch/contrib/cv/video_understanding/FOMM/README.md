# FOMM模型-推理指导


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

FOMM模型最早是Aliaksandr Siarohin等人在发表的《First Order Motion Model for Image Animation》一文中提到的用于图像动画化（image animation）的模型。图像动画化任务是指给定一张原图片和一个驱动视频，通过视频合成，生成主角为原图片，而动画效果和驱动视频一样的视频。以往的视频合成往往依赖于预训练模型来提取特定于对象的表示，而这些预训练模型是使用昂贵的真实数据注释构建的，并且通常不适用于任意对象类别。而FOMM的提出很好的解决了这个问题。


- 参考实现：

```
url=https://github.com/AliaksandrSiarohin/first-order-model.git
branch=master
commit_id=0b04120d53abadf50cc283349e1bb8c6ae693ebc
model_name=FOMM
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

- kp detector输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | input    | RGB_FP32 | 1 x 3 x 256x 256 | NCHW         |


- kp detector输出数据

  | 输出数据 | 大小           | 数据类型 | 数据排布格式 |
  | -------- | -------------- | -------- | ------------ |
  | value    | 1 x 10 x 2     | FLOAT32  | NCH          |
  | jacobian | 1 x 10 x 2 x 2 | FLOAT32  | NCHW         |

- generator输入数据

  | 输入数据    | 数据类型 | 大小                              | 数据排布格式      |
  | ----------- | -------- | --------------------------------- | ----------------- |
  | source_imgs | RGB_FP32 | 1 x 3 x 256 x 256                 | NCHW              |
  | kp_driving  | RGB_FP32 | 1 x 10 x 2；<br/>1 x 10 x 2 x 2； | NCHW              |
  | kp_source   | RGB_FP32 | 1 x 10 x 2；<br/>1 x 10 x 2 x 2； | NCH；<br />NCHW； |


- generator输出数据

  | 输出数据 | 数据类型 | 大小                                                         | 数据排布格式 |
  | -------- | -------- | ------------------------------------------------------------ | ------------ |
  | out      | FLOAT32  | 1 x 11x 64 x 64；<br />1 x 11 x 3 x 64 x 64；<br />1 x 1 x 64 x 64；<br />1 x 3 x 256 x 256；<br />1 x 3 x 256 x 256； | ND           |



# 推理环境准备\[所有版本\]

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.11.0  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手

## 获取源码

1. 获取源码。

   在`FOMM/`下执行如下命令：

   ```shell
   git clone https://github.com/AliaksandrSiarohin/first-order-model.git
   ```

2. 整理目录

   由于在进行模型转换的时候，FOMM中使用的某些算子是onnx所不支持的，所以需要对FOMM中的某些代码进行一定的改动，所以在PR中的FOMM/my_script/目录下我们提供了修改后的python程序，用以替换github源码仓拉去下来的源码中的部分程序。除此之外，该目录下还有一些其他脚本和环境依赖requirements文件。所以我们需要先整理一下项目目录，以满足后续开发使用需求。

   执行以下命令整理项目目录：

   ```shell
   cd first-order-model
   mv * ../
   cd ..
   rm -rf first-order-model
   cd my_script
   mv modules/generator.py ../modules/generator.py
   mv modules/dense_motion.py ../modules/dense_motion.py
   
   mv reconstruction.py ../
   mv requirements.txt ../
   mv taichi-256.yaml ../config/
   mv logger.py ../
   
   cd ..
   ```

3. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

4. 安装其他依赖

   在`FOMM/`下安装以下依赖：

   maskrcnn-benchmark：

   ```shell
   git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
   cd maskrcnn-benchmark
   python setup.py install
   cd ..
   ```

   pose-evaluation：

   首先下载pose_model.pth，保存到`FOMM/`下，下载链接为：https://yadi.sk/d/0L-PgAaGRKgkJA

   然后在`FOMM/`下依次执行如下命令

   ```shell
   git clone --recursive https://github.com/AliaksandrSiarohin/pose-evaluation
   mkdir pose-evaluation/pose_estimation/network/weight/
   mv pose_model.pth pose-evaluation/pose_estimation/network/weight/
   ```

   ascend tools：

   在`FOMM/`下执行如下建议安装ascend tools推理工具包：

   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。
   
   magiconnx:
   
   在`FOMM/`下依次执行如下命令安装该包：
   
   ```shell
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
   cd MagicONNX
   pip install .
   cd ..
   ```

## 准备数据集

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   推理使用的 <u>**taichi**</u> 数据集的下载地址如下：

   https://pan.baidu.com/s/1yBTPpa5ZrEmwQHNVmEdpow

   提取码：1234

   下载下来以后放在`FOMM/data/`目录下。

   保存好后目录结构大致如下：

   ```
   data
      |-- taichi
      `-- |-- test
          |   `-- 0Q914by5A98#010440#010764.mp4
          |   `-- 8hLvlQrXI6U#007700#007984.mp4
          |   `-- 8hLvlQrXI6U#008247#008392.mp4
          |   `-- ...
          `-- train
   ```

2. 获取权重文件。

   下载链接：https://drive.google.com/open?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH

   打开链接后只下载云盘里的taichi-cpk.pth.tar即可

   然后在`FOMM/`下执行如下命令：

   ```shell
   mkdir checkpoint
   ```

   将下载后的权重文件保存在`FOMM/checkpoint/`下

   ```
   checkpoint
      |-- taichi-cpk.pth.tar
      |-- ...
   ```

3. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行`FOMM_preprocess.py`脚本，完成数据预处理。

   ```shell
   python FOMM_preprocess.py --config config/taichi-256.yaml --checkpoint checkpoint/taichi-cpk.pth.tar --data_type npy --out_dir pre_data/
   ```

   参数说明

   * config：配置文件的相对路径

   * checkpoint：检查点文件（.pth.tar文件）的相对路径
   * data_type：输出的数据的格式，npy或bin（建议使用npy格式）
   * out_dir：输出的数据保存的位置，默认为”./pre_data/“。（建议使用默认值）

   运行成功后，分别在**FOMM/pre_data/driving/和FOMM/pre_data/source/**两个文件夹下生成对应的npy数据文件

   ```
   pre_data
      |-- driving
      `-- |-- 0.npy
      `-- |-- 1.npy
      `-- |-- 2.npy
      `-- |-- ...(共64196个npy文件)
      |-- source
      `-- |-- 0.npy
      `-- |-- 1.npy
      `-- |-- 2.npy
      `-- |-- ...(共64196个npy文件)
   ```


## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   （1）导出onnx文件。

   使用FOMM_pth2onnx.py导出onnx文件。

   在`FOMM/`下执行如下命令：

   ```shell
   mkdir taichi-onnx
   python FOMM_pth2onnx.py --config config/taichi-256.yaml --checkpoint checkpoint/taichi-cpk.pth.tar --outdir taichi-onnx --genname taichi-gen-bs1 --kpname taichi-kp-bs1
   ```

   参数说明：

   * config：配置文件的相对路径；

   * checkpoint：权重文件存放路径；

   * outdir：onnx模型的保存目录；

   * genname：导出的generator模型的文件名；

   * kpname：导出的kp detector模型的文件名；

   获得`./taichi-onnx/taichi-kp-bs1.onnx`、`./taichi-onnx/taichi-gen-bs1.onnx`文件。

   ```
   taichi-onnx
      |-- taichi-kp-bs1.onnx
      |-- taichi-gen-bs1.onnx
   ```

   然后需要对输出的taichi-gen-bs1.onnx进行算子优化，依次执行如下命令即可：

   ```shell
   mv expand_int32.py taichi-onnx/
   cd taichi-onnx
   python expand_int32.py
   cd ..
   ```

   （2）使用ATC工具将ONNX模型转OM模型。

   首先在`FOMM/`下创建`./taichi-onnx/oms`文件夹：

   ```shell
   mkdir taichi-onnx/oms
   ```

   a. 配置环境变量。

   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   > **说明：** 
   >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   b. 执行命令查看芯片名称（$\{chip\_name\}）。

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
   | 0       310P3     | OK              | 16.0         50                1236 / 1236           |
   | 0       0         | 0000:86:00.0    | 0            4066 / 21534                            |
   +===================+=================+======================================================+
   ```

   c. 执行ATC命令。

   generator

   ```shell
   atc --framework=5 --model=taichi-onnx/new_expand_taichi_gen_bs1.onnx --output=taichi-onnx/oms/taichi-gen-bs1 --input_format=NCHW --input_shape="source_imgs:1,3,256,256;kp_driving_value:1,10,2;kp_driving_jac:1,10,2,2;kp_source_value:1,10,2;kp_source_jac:1,10,2,2" --log=debug --soc_version=Ascend${chip_name} --buffer_optimize=off_optimize
   ```

   kp detector

   ```shell
   atc --framework=5 --model=taichi-onnx/taichi-kp-bs1.onnx --output=taichi-onnx/oms/taichi-kp-bs1 --input_format=NCHW --input_shape="input:1,3,256,256" --log=debug --soc_version=Ascend${chip_name} --buffer_optimize=off_optimize
   ```

   参数说明：

   * model：为ONNX模型文件。

   -   framework：5代表ONNX模型。
   -   output：输出的OM模型。
   -   input\_format：输入数据的格式。
   -   input\_shape：输入数据的shape。
   -   log：日志级别。
   -   soc\_version：处理器型号。

   ```
   taichi-onnx
      |-- taichi-kp-bs1.onnx
      |-- taichi-gen-bs1.onnx
      |-- oms
      |   `-- taichi-kp-bs1.om
      |   `-- taichi-gen-bs1.om
   ```

2. 开始推理验证。<u>***根据实际推理工具编写***</u>

   a. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

   b.  执行推理。

   首先，此处对推理流程做大致解释：

   FOMM模型推理涉及到里面的两个子模型kp detector和generator，并且是围绕这两个子模型进行的。其中generator的输入如前面”输入输出数据“一章中所介绍的，有三个，分别是source_imgs（预处理后的原始图片数据），**kp driving（将预处理后的driving数据输入kp detector后得到的数据）**以及**kp source（将预处理后的source_imgs数据输入kp detector后得到的数据）**。

   所以接下来的模型推理需要先对kp detector推理两次，得到kp driving和kp source两个输出数据，然后才能对generator进行推理。

   另外，由于模型中**kp detector输出的数据是python的字典类型（包含"value"和"jacobian"两个键）**，所以在对kp detector进行了两次推理得到kp driving和kp source后还要执行一个数据整理脚本将数据整理保存，方便后面对generator推理时使用。

   综上所述，首先对kp detector进行推理，依次运行如下命令即可：

   * kp driving:

   ```shell
   mkdir infer_out
   python -m ais_bench --model taichi-onnx/oms/taichi-kp-bs1.om --input pre_data/driving/ --output infer_out/ --outfmt NPY --output_dirname kpd
   ```
   
   * kp source
   
   ```shell
   python -m ais_bench --model taichi-onnx/oms/taichi-kp-bs1.om --input pre_data/source/ --output infer_out/ --outfmt NPY --output_dirname kps
   ```
   
   参数说明：
   
   * model：需要进行推理的om模型
   * input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据
   * output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
   * output_dirname：推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中
   * outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”
   
   ```
   infer_out
      |-- kpd
      `-- |-- 0_0.npy
      `-- |-- 0_1.npy
      `-- |-- 1_0.npy
      `-- |-- 1_1.npy
      `-- |-- ...(共128392个npy文件)
      |-- kps
      `-- |-- 0_0.npy
      `-- |-- 0_1.npy
      `-- |-- 1_0.npy
      `-- |-- 1_1.npy
      `-- |-- ...(共128392个npy文件)
   ```
   
   执行完上述推理后，为了方便之后推理使用，执行如下命令整理输出的数据：
   
   ```shell
   python apart_kp_out.py --type npy --data_root infer_out --driving_dir kpd --source_dir kps
   ```
   
   执行完毕后，将会在--data_root下生成4四个文件夹：kpdv、kpdj、kpsv、kpsj，分别用于存放--driving_dir中的value、jacobian和--source_dir中的value和jacobian。
   
   参数说明：
   
   * type：数据文件的格式，这次npy和bin，建议使用npy；
   
   * data_root：数据整理后数据保存的目录，对应前面推理命令中的--output参数，默认为infer_out；
   
   * driving_dir：kp driving（上面第一个推理命令得到的输出）保存的文件夹，对应上面第一个推理命令的--output_dirname参数；
   * source_dir：kp source（上面第二个推理命令得到的输出）保存的文件夹，对应上面第二个推理命令的--output_dirname参数；
   
   ```
   infer_out
      |-- kpd
      `-- |-- 0_0.npy
      `-- |-- 0_1.npy
      `-- |-- 1_0.npy
      `-- |-- 1_1.npy
      `-- |-- ...(共128392个npy文件)
      |-- kps
      `-- |-- 0_0.npy
      `-- |-- 0_1.npy
      `-- |-- 1_0.npy
      `-- |-- 1_1.npy
      `-- |-- ...(共128392个npy文件)
      |-- kpdv
      `-- |-- 0.npy
      `-- |-- 1.npy
      `-- |-- 2.npy
      `-- |-- ...(共64196个npy文件)
      |-- kpdj
      `-- |-- 0.npy
      `-- |-- 1.npy
      `-- |-- 2.npy
      `-- |-- ...(共64196个npy文件)
      |-- kpsv
      `-- |-- 0.npy
      `-- |-- 1.npy
      `-- |-- 2.npy
      `-- |-- ...(共64196个npy文件)
      |-- kpsj
      `-- |-- 0.npy
      `-- |-- 1.npy
      `-- |-- 2.npy
      `-- |-- ...(共64196个npy文件)
   ```
   
   现在，一切数据都准备好了，执行如下命令，推理generator模型：
   
   ```shell
   mkdir infer_out/out/
   python -m ais_bench --model taichi-onnx/oms/taichi-gen-bs1.om --input pre_data/source/,infer_out/kpdv/,infer_out/kpdj/,infer_out/kpsv/,infer_out/kpsj/ --output infer_out/ --outfmt NPY --output_dirname out/
   ```
   
   至此，模型推理部分就完成了。
   
   ```
   infer_out
      |-- out
      `-- |-- 0_0.npy
      `-- |-- 0_1.npy
      `-- |-- 0_2.npy
      `-- |-- 0_3.npy
      `-- |-- 0_4.npy
      `-- |-- 1_0.npy
      `-- |-- ...(共320980个npy文件)
      |-- ...
   ```
   
   c.  精度验证。
   
   验证模型的精度指标，依次执行如下命令即可
   
   ```shelll
   python FOMM_reconstruction.py --config config/taichi-256.yaml --checkpoint checkpoint/taichi-cpk.pth.tar --data_type npy
   ```
   
   参数说明：
   
   * config：配置文件路径；
   
   * checkpoint：.pth.tar权重文件保存路径；
   
   * data_type：使用的数据的格式，建议使用npy；
   
   * data_dir：推理输出的数据保存的目录，对应前面推理命令中的--output参数，默认为infer_out，建议使用默认值；
   
   * pre_data：预处理后数据保存的目录，对应前面预处理命令中的--out_dir参数，默认为pre_data，建议使用默认值。
   
   ```shell
   cd pose-evaluation/
   mv ../my_script/coco_eval.py pose_estimation/evaluate/
   mv ../my_script/extract.py ./
   python extract.py --in_folder ../data/taichi/test/ --out_file pose_gt.pkl --is_video --type body_pose --image_shape 256,256
   python extract.py --in_folder ../checkpoint/reconstruction/png --out_file pose_gen.pkl --is_video --type body_pose --image_shape 256,256
   ```
   
   参数说明：
   
   * in_folder：输入的视频或图片的保存目录；
   
   * out_file：输出的.pkl文件的文件名；
   
   * type：使用的函数的类型；
   
   * image_shape：帧图片的shape。
   
   最后运行如下命令，即可得到两个精度指标。
   
   ```shell
   python cmp_with_missing.py pose_gt.pkl pose_gen.pkl
   ```


# 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P | 1 | taichi | ADK：6.80694477；<br />MKR：0.03592159 | kp detector：860.282664209366<br />generator：14.380611007559 |

PS：该模型只支持bs1

