\# CycleGAN模型端到端推理指导

\-  [1 模型概述](#1-模型概述)

​    \-  [1.1 论文地址](#11-论文地址)

​    \-  [1.2 代码地址](#12-代码地址)

\-  [2 环境说明](#2-环境说明)

​    \-  [2.1 深度学习框架](#21-深度学习框架)

​    \-  [2.2 python第三方库](#22-python第三方库)

\-  [3 模型转换](#3-模型转换)

​    \-  [3.1 pth转onnx模型](#31-pth转onnx模型)

​    \-  [3.2 onnx转om模型](#32-onnx转om模型)

\-  [4 数据集预处理](#4-数据集预处理)

​    \-  [4.1 数据集获取](#41-数据集获取)

​    \-  [4.2 数据集预处理](#42-数据集预处理)

​    \-  [4.3 生成数据集信息文件](#43-生成数据集信息文件)

\-  [5 离线推理](#5-离线推理)

​    \-  [5.1 benchmark工具概述](#51-benchmark工具概述)

​    \-  [5.2 离线推理](#52-离线推理)

\-  [6 精度对比](#6-精度对比)

​    \-  [6.1 离线推理精度统计](#61-离线推理精度统计)

​    \-  [6.2 在线推理精度](#62-在线推理精度)

​    \-  [6.3 精度对比](#63-精度对比)

\-  [7 性能对比](#7-性能对比)

​    \-  [7.1 npu性能数据](#71-npu性能数据)

​    \-  [7.2 性能优化](#73-性能优化)

​      \- [7.2.1 优化TransData，修改five_2_four.py](#731-优化TransData，修改five_2_four.py)

\## 1 模型概述

 

\-  **[论文地址](#11-论文地址)** 

 

\-  **[代码地址](#12-代码地址)** 

 

\### 1.1 论文地址

 

[CycleGAN论文]( https://arxiv.org/pdf/1703.10593v7.pdf) 

我们专注于本文中风格转换中的地图转换。它通过一种无监督的少样本的学习方式，能够实现航拍地图和卫星地图之间的相互转换。

\### 1.2 代码地址

[CycleGAN代码]( https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 

branch:master 

commit_id:略

备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交

 

\## 2 环境说明

 

\-  **[深度学习框架](#21-深度学习框架)** 

 

\-  **[python第三方库](#22-python第三方库)** 

 

\### 2.1 深度学习框架

\```

```
CANN 5.0.2.alpha003

torch == 1.5.0

torchvision == 0.9.0

onnx==1.7.0

onnx-simplifier==0.3.6

onnxconverter-common==1.6.1

onnxoptimizer==0.2.6

onnxruntime==1.6.0

tensorboard==1.15.0

tensorflow==1.15.0

tensorflow-estimator ==1.15.1

termcolor==1.1.0
```

\```

 

\### 2.2 python第三方库

\```

```
numpy == 1.16.6

Pillow == 8.2.0

opencv-python == 4.5.2.52

sympy == 1.4

decorator == 4.4.2

requests == 2.22.0

tqdm == 4.61.0

PyYAML == 5.4.1
```

\```

 

**说明：** 

\>  X86架构：pytorch torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3 install 包名 安装 

\>  Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3 install 包名 安装

 

\## 3 模型转换

 

\-  **[pth转onnx模型](#31-pth转onnx模型)** 

 

\-  **[onnx转om模型](#32-onnx转om模型)** 

 

\### 3.1 pth转onnx模型

1.下载开源模型代码，安装必要的依赖库，并修改模型代码后安装  

\```

```
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
pip3  install -r requirements.txt
```

\```

 

2.下载pth权重文件 

 

\- [官方CycleGAN pth权重文件](http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/) 

\- [获取A800-9000训练的pth文件,该链接为百度网盘链接，提取码为：1234](https://pan.baidu.com/s/1YqHkce2wUw-W8_VY9dYD_w))

3.编写pth2onnx脚本*CycleGAN_onnx_export.py*

 **说明：** 

\>注意目前ATC支持的onnx算子版本为11

 

4.执行*CycleGAN_onnx_export.py*脚本，生成onnx模型文件 

\```

```
python3 CycleGAN_onnx_export.py \

--model_ga_path=./checkpoints/maps_cycle_gan/latest_net_G_A.pth\

--model_gb_path=./checkpoints/maps_cycle_gan/latest_net_G_B.pth\

--onnx_path=./onnxmodel/      \

--model_ga_onnx_name=model_Ga.onnx       \

--model_gb_onnx_name=model_Gb.onnx   \
```

\```

 **模型转换要点：** 

\- 开源仓中的生成器采用的padding类型为ReflectionPad2d，由于在转om格式模型的时候，会出现算子不兼容问题导致om模型转换失败，这里我们将改padding类型替换为ZeroPad2d。如果您任然坚持使用ReflectionPad2d，请在转换Onnx格式后运行

 ' ' '

```
python3 CycleGAN_ReflectpadDeal.py  \

--onnx_path=./onnxmodel/            \

--model_ga_onnx_name=model_Ga.onnx  \

--model_gb_onnx_name=model_Gb.onnx   \
```

' ' '

该脚本会将ReflectionPad2d中的属性替换为constant，这样做的结果会导致模型执行推理时会出现边缘模糊，详情请见issue链接https://e.gitee.com/HUAWEI-ASCEND/issues/list?issue=I4467L#note_6141945

 

\### 3.2 onnx转om模型

 

1.设置环境变量

\```

```
source env.sh
```

\```

\- 根据实际情况修改env.sh中的install_path=/usr/local/Ascend/ascend-toolkit/latest变量 

\- 执行脚本前先执行指令 dos2unix *

 

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.1 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

\```

```
atc --framework=5 --model=./onnxmodel/model_Ga.onnx --output=Cons_Ga_aipp512_b0_bs1 --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310 --insert_op_conf=aipp_CycleGAN_pth.config

atc --framework=5 --model=./onnxmodel/model_Gb.onnx --output=Cons_Gb_aipp512_b0_bs1 --input_format=NCHW --input_shape="img_maps_sat:1,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310 --insert_op_conf=aipp_CycleGAN_pth.config
```

\```

\- 说明 

  \- input_shape参数可通过Netron工具查看输入节点的名称和shape, 与pth转onnx步骤中的参数一致 

  \- out_nodes为指定输出节点, 通过Netron可以看到onnx文件有四个输出, 以自测转换的onnx为例 

  如果在转onnx时使用的不是默认路径，请将—model中的参数设置为onnx格式模型所在的路径

 

 

\## 4 数据集预处理

 

\-  **[数据集获取](#41-数据集获取)** 

 

\-  **[数据集预处理](#42-数据集预处理)** 

 

\-  **[生成数据集信息文件](#43-生成数据集信息文件)** 

 

\### 4.1 数据集获取

该模型使用[maps数据集](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/maps.zip)的testA和testB各1098张验证集进行测试，因为航拍地图和卫星地图之间的相互转换的两个生成器模型结构一样，这里我们只需要保证其中一个生辰器精度和性能跟上就行，这里我们以model_Ga.onnx和testA为推理的模型和测试数据集。

 

\### 4.2 数据集预处理

1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

\```

```
python3 gen_dataset_info.py \ 

--src_path_testA=./datasets/maps/testA/     \ 

--save_pathTestA_dst=datasetsDst/maps/testA/  \

--dataTestA_infoName=testA_prep.info     \

--src_path_testB=./datasets/maps/testB/     \

--save_pathTestB_dst=./datasetsDst/maps/testB/  \

--dataTestB_infoName=testB_prep.info
```

' ' '

\## 5 离线推理

 

\-  **[benchmark工具概述](#51-benchmark工具概述)** 

 

\-  **[离线推理](#52-离线推理)** 

 

\### 5.1 benchmark工具概述

 

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN 5.0.1 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

\### 5.2 离线推理

1.设置环境变量

\```

source env.sh

\```

2.执行离线推理 

\- benchmark工具区分arm64和x86_64, 对应分别为./benchmark.aarch64和./benchmark.x86_64, 示例中均以x86_64环境为例

\- 将benchmark工具去相应路径获取后放到env.sh同级目录下，加上执行权限chmod +x benchmark.XX  

'''

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=Cons_Ga_aipp512_b0_bs1.om -input_text_path=testA_prep.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=true

./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=Cons_Gb_aipp512_b0_bs1.om -input_text_path=testB_prep.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=true
```

输出结果默认保存在当前目录result/dumpOutput_devicex，每个输入对应的输出对应一个_x.bin文件。

'''

\## 6 精度对比

\### 6.1 离线推理精度统计

由于该模型的精度在论文中是由人眼分辨，所以这里我们那Onnx和om模型输出的平均余弦相似度来替代精度，只需要保证Onnx格式模型的效果和论文中的一致并且om和onnx格式模型的余弦相似度在99%左右就精度达标。执行eval_acc_py.py脚本计算平均余弦相似度 :

\```

```

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=Cons_Ga_aipp512_b0_bs1.om -input_text_path=testA_prep.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=true  #如果已经执行这一步请忽略
python3 eval_acc.py \
--dataroot=./datasets/maps\
--npu_bin_file=./result/dumpOutput_device0/
```

\```

\### 6.2精度对比

![1](C:\Users\Administrator\Desktop\1.png)

将得到的om离线模型推理精度与在线推理精度对比，推理精度与在线推理精度一致，精度达标。 

 **精度调试:** 

使用onnxruntime测试onnx离线推理精度与om一致。

\## 7 性能对比

\-  **[npu性能数据](#71-npu性能数据)** 

\-  **[性能优化](#73-性能优化)** 

\### 7.1 npu性能数据

这里用batch1和batch16做示例  

 

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，模型的测试脚本使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准。  

 

1.benchmark工具在整个数据集上推理获得性能数据 

 

以batch1为例，benchmark工具在整个数据集上推理,执行下面命令。

```
atc --framework=5 --model=./onnxmodel/model_Ga.onnx --output=Cons_Ga_aipp512_b0_bs1 --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310 --insert_op_conf=aipp_CycleGAN_pth.config #如果已经转换，请忽略
python3.7 gen_dataset_info.py  #如果这一步已经执行，可直接执行下一步推理
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=Cons_Ga_aipp512_b0_bs1.om -input_text_path=testA_prep.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=true
```

\```

 ![输入图片说明](https://images.gitee.com/uploads/images/2021/0914/121624_f45173ef_9486012.png "屏幕截图.png")



Interface throughputRate: 10.7，10.7乘以4，是310单卡吞吐率 

 \```

2.benchmark纯推理功能测得性能数据 

 

batch1的性能：

 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

\```

```
./benchmark.x86_64 -round=20 -om_path=Cons_Ga_aipp512_b0_bs1.om -device_id=0 -batch_size=1
```

```

执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果

 ![输入图片说明](https://images.gitee.com/uploads/images/2021/0914/121641_4ed82b8d_9486012.png "屏幕截图.png")
```


Batch16的性能：

```
./benchmark.x86_64 -round=20 -om_path=model_Ga-b0_bs16.om -device_id=1 -batch_size=16
```

![输入图片说明](https://images.gitee.com/uploads/images/2021/0914/121659_6331aa3d_9486012.png "屏幕截图.png")

\### 7.2 性能优化

```
**性能优化** 

\- profiling性能分析方法 

​       CANN C20及以后的版本profiling使用方法

新建/home/zlz/CycleGan_deal/perProblem_detec/run文件，内容如下：

```
```
# /usr/local/Ascend/ascend-toolkit/  /usr/local/Ascend/ascend-toolkit/
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
./benchmark.x86_64 -round=20 -om_path=/home/zlz/cyclegan/model_Ga1-b0_bs16.om -device_id=0 -batch_size=16
```

然后执行如下命令：
```
chmod 777 /home/zlz/CycleGan_deal/perProblem_detec/run
cd /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/toolkit/tools/profiler/bin
./msprof --output=/home/zlz/CycleGan_deal/perProblem_detec/perPro/ --application=/home/zlz/CycleGan_deal/perProblem_detec/run --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on --sys-io-profiling=on --dvpp-profiling=on
cd /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/toolkit/tools/profiler/profiler_tool/analysis/msprof/
# 生成的profiling目录
python3.7 msprof.py import -dir/home/zlz/CycleGan_deal/perProblem_detec/perPro/    
python3.7 msprof.py export summary -dir /home/zlz/CycleGan_deal/perProblem_detec/perPro/ 
#生成的profiling目录 --iteration-id 1
python3.7 msprof.py export timeline -dir /home/zlz/CycleGan_deal/perProblem_detec/perPro/ 

```
目录

\- 性能调优测试版本：CANN 5.0.2.alpha003

\- 性能优化过程主要对trans_Data算子进行优化，结合profiling分析，性能有提升:

\#### 7.3.1 five_2_four.py优化方法 

 在环境变量env.sh中export install_path=/usr/local/Ascend/ascend-toolkit/latest路径下查找five_2_four.py文件，路径一般为

\```  

/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/five_2_four.py

\```

修改five_2_four.py文件，将TransData算子的output shape加入five_2_four函数行中，示例如下：

\```

    ...
    from impl import trans_data_negative_target_ntc
    
    @util.check_input_type(dict, dict, str, str, str)
    
    def five_2_four(src, dst, src_format, dst_format, kernel_name='five_2_four'): 
    elif dst_format.lower() == "nhwc" and dst_shape in [[10000, 63, 63, 1], [10000, 127, 127, 1], [16, 19, 19, 486],
    
                             [16, 10, 10, 486], [16, 38, 38, 324], [16, 5, 5, 486],
    
                            [16, 3, 3, 324], [8, 19, 19, 486], [8, 10, 10, 486],
    
                            [8, 38, 38, 324], [8, 5, 5, 486], [8, 3, 3, 324],
    
                            [100, 28, 28, 91]]:
    
    trans_data_negative_target_tc.trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)

  elif dst_format.lower() == "nchw" and dst_shape in [[2560, 512, 4, 26], [2560, 512, 1, 26], [2560, 256, 8, 25],

                            [16, 240, 7, 7], [16, 120, 14, 14], [1,19,1024,2048], [4,19,1024,2048]]:
    
    print("=================================")
    
    print("ntc dst shape:", dst_shape)
    
    print("=================================")
    
    trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
    ...

\```

\- 不同的batch_size,添加的shape不一样，shape大小为[*，19,256,256 ] ,以本模型为例，只测试batch1和batch16,因此添加的shape为[1,19,256,256],[4,19,256,256]

修改完成后，重新转换生成om文件，atc转换过程会打印添加的日志，如下： 

\```![输入图片说明](https://images.gitee.com/uploads/images/2021/0914/121715_d94592ad_9486012.png "屏幕截图.png")

 \```

纯推理测试结果：

\```

bs1:

 ![输入图片说明](https://images.gitee.com/uploads/images/2021/0914/121721_50c95bdd_9486012.png "屏幕截图.png")

Bs16:

![输入图片说明](https://images.gitee.com/uploads/images/2021/0914/122022_a16e9ff5_9486012.png "屏幕截图.png")

\```

 

用生成的om文件做精度后处理，测得bs1和bs16与之前的Onnx模型做余弦相似度高于99%，精度无损失、

\```

\#### 7.3.1 总结

优化方案共包括五种： 

（1）优化TransData，修改five_2_four.py 

（2）输出节点由float32改为float16 

（3）模型中Resize节点的mode由双线性为最近邻 

（4）将PadV3D进行算子融合 

（5）优化FrameworkOP框架 

由于在蓝区测试的版本CANN 5.0.2.alpha003中，已经实现了PadV3D算子融合，因此测试过程默认已经优化。同时方案（5）暂时无法实现，因此也无法比对性能。

结论：

\- 因为关键算子性能差，性能暂时无法达标。

\- 最终精度测试，Om模型输出效果达到论文效果，转Om后无精度损失。