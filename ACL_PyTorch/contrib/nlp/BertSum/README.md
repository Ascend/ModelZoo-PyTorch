

# BertSum Onnx模型端到端推理指导

- 1 模型概述
  - [1.1 代码地址](https://gitee.com/kghhkhkljl/pyramidbox.git)
- 2 环境说明
  - [2.1 深度学习框架](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#21-深度学习框架)
  - [2.2 python第三方库](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#22-python第三方库)
- 3 模型转换
  - [3.1 pth转onnx模型](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#31-pth转onnx模型)
  - [3.2 onnx转om模型](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#32-onnx转om模型)
- 4 数据集预处理
  - [4.1 数据集获取](https://www.graviti.cn/open-datasets/WIDER_FACE)
  - [4.2 数据集预处理](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#42-数据集预处理)
  - [4.3 生成数据集信息文件](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#43-生成数据集信息文件)
- 5 离线推理
  - [5.1 benchmark工具概述](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#51-benchmark工具概述)
  - [5.2 离线推理](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#52-离线推理)
- 6 精度对比
  - [6.1 离线推理精度统计](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#61-离线推理精度统计)
  - [6.2 开源精度](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#62-开源精度)
  - [6.3 精度对比](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#63-精度对比)
- 7 性能对比
  - [7.1 npu性能数据](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#71-npu性能数据)
  - [7.2 T4性能数据](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#72-T4性能数据)
  - [7.3 性能对比](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#73-性能对比)

## 1 模型概述

- **论文地址**
- **代码地址**

### 1.1 论文地址

[Bertsum论文](https://arxiv.org/abs/1803.07737)

### 1.2 代码地址

https://github.com/nlpyang/BertSum.git

## 2 环境说明

- **深度学习框架**
- **python第三方库**

### 2.1 深度学习框架

```
python3.7.5
CANN 5.0.3

pytorch >= 1.5.0
torchvision >= 0.10.0
onnx >= 1.7.0

说明：若是在conda环境下，直接采用python，不用python3.7
```

### 2.2 python第三方库

```
torch==1.7.1
tensorboardX==2.4.1
pyrouge==0.1.3
pytorch-pretrained-bert==0.6.2
onnx-simplifier==0.3.6
```

### **2.3 环境配置**

ROUGE配置参考博客：

[(10条消息) Ubuntu安装配置ROUGE_BigSea-CSDN博客](https://blog.csdn.net/Hay54/article/details/78744912)

pyrouge配置参考博客：

[(10条消息) 在Ubuntu下配置pyrouge_MerryCao的博客-CSDN博客](https://blog.csdn.net/MerryCao/article/details/49174283)

## 3 模型转换

- **pth转onnx模型**
- **onnx转om模型**

### 3.1 pth转onnx模型

1.拉取代码仓库 （因为使用了开源代码模块，所以需要git clone一下）

```shell
git clone https://github.com/nlpyang/BertSum.git
```

克隆下来源代码并解压，将pr中的代码放到解压之后的BertSum/src目录下面并对BertSum/src/models/data_loder.py进行一个更改： 

将31行的mask=1-(src==0)修改为mask=~(src==0) 将35行的mask=1-(clss==-1)修改为mask=~(clss==-1)

2.下载pth权重文件

权重文件默认存放在**/home/BertSum/src**目录下

3.使用pth2onnx.py进行onnx的转换

```
方法一：cd /home/BertSum/src/test
bash pth2onnx.sh
方法二：cd /home/BertSum/src
python BertSum-pth2onnx.py -mode test -bert_data_path ../bert_data/cnndm -model_path MODEL_PATH -visible_gpus -1 -gpu_ranks 0 -batch_size 1 -log_file LOG_FILE -result_path RESULT_PATH -test_all -block_trigram true -onnx_path bertsum_13000_9_bs1.onnx -path model_step_13000.pt
```

获得bertsum_13000_9_bs1.onnx文件

方法二种的-bert_data_path是数据集所在目录，-batch_size需设置为1，-onnx_path是onnx输出文件

### 3.2 onnx模型简化

由于存在expand算子导致转om不成功，所以需要使用onnx简化工具对onnx进行简化

使用pth2onnx.py进行onnx的转换

```
方法一：cd /home/BertSum/src/test
bash simplify.sh
方法二：cd /home/BertSum/src
python -m onnxsim ./bertsum_13000_9_bs1.onnx ./bertsum_13000_9_sim_bs1.onnx
```

获得bertsum_13000_9_sim_bs1.onnx文件

### 3.3 onnx简化模型转om模型

1.设置环境变量

```
source atc.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.3 开发辅助工具指南 (推理) 01

```
方法一：cd /home/BertSum/src/test
bash onnxToom.sh 
方法二：cd /home/BertSum/src
atc --input_format=ND --framework=5 --model=./bertsum_13000_9_sim_bs1.onnx --input_shape="src:1,512;segs:1,512;clss:1,37;mask:1,512;mask_cls:1,37" --output=bertsum_13000_9_sim_bs1  \
--log=info --soc_version=Ascend310 --precision_mode=allow_mix_precision \
--modify_mixlist=ops_info.json
```

方法二中的model是onnx模型的名字，input_shape是paper的shape，output为输出om的名字，--precision_mode表示采用混合精度

## 4 数据集预处理

- **数据集获取**
- **数据集预处理**
- **生成数据集信息文件**

### 4.1 数据集获取

参考原代码仓

### 4.2 数据集预处理

1.预处理脚本BertSum_pth_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
方法一：cd /home/BertSum/src/test
bash pre_deal.sh
方法二：cd /home/BertSum/src
python BertSum_pth_preprocess.py -mode test -bert_data_path ../bert_data/cnndm -model_path MODEL_PATH -visible_gpus -1 -gpu_ranks 0 -batch_size 600 -log_file LOG_FILE -result_path RESULT_PATH -test_all -block_trigram true
```

-bert_data_path是数据集所在目录，后面的参数是固定的。

### 5 离线推理

- **msame工具**
- **离线推理**

### 5.1 msame工具

获取msame工具（https://gitee.com/ascend/tools/tree/master/msame），并将得到的msame工具放在/home/BertSum-master/src下

### 5.2 离线推理

1.执行离线推理

benchmark工具暂不支持多输入，因此改用msame，首先要source环境变量

```
source env.sh
```

2.使用msame将onnx模型转换为om模型文件，工具使用方法可以参考CANN 

然后运行如下命令：

```
方法一：cd /home/BertSum/src/test
bash infer.sh
方法二：cd /home/BertSum/src
./msame --model "./bertsum_13000_9_sim_bs1_1.om" --input "./pre_data/src,./pre_data/segs,./pre_data/clss,./pre_data/mask,./pre_data/mask_cls" --output "./result" --outfmt bin
./msame --model "./bertsum_13000_9_sim_bs1_1.om" --input "./pre_data_1/src,./pre_data_1/segs,./pre_data_1/clss,./pre_data_1/mask,./pre_data_1/mask_cls" --output "./result" --outfmt bin
```

要采用msema工具推理两次，因为有些paper的shape第一维为2，所以分两次进行推理。pre_data下存放的是shape为第一维为1的所有预处理之后的数据以及shape为2的部分预处理得到的数据。shape为2的另一部分数据存放在pre_data_1下面。--model是om文件，--input是预处理之后文件所在目录，--output为输出bin文件所在目录，--outfmt代表输出bin文件*。*

输出的bin文件在/home/BertSum-master/src/result目录下，此目录下会存在两个文件，将其中一个时间小的命名为result_1,将另一个时间大的命名为result_2。

## 6 精度对比

- **离线推理精度**
- **开源精度**
- **精度对比**

### 6.1 离线推理精度统计

1.后处理

```
cd /home/BertSum/src
python BertSum_pth_postprocess.py  -visible_gpus -1 -gpu_ranks 0 -batch_size 600 -log_file LOG_FILE -result_path RESULT_PATH -test_all -block_trigram true -path_1 ./result/result_1 -path_2 ./result/result_2
```

```
 -path_1是推理得到的文件result_1,-path_2是推理得到的result_2
 自验报告
  # 第X次验收测试   
  # 验收结果 OK 
  # 验收环境: A + K / CANN 5.0.3
  # 关联issue: 
  
  # pth是否能正确转换为om
  bash test/onnx2om.sh
  # 验收结果： OK 
  # 备注： 成功生成om，无运行报错，报错日志xx 等
  
  # 精度数据是否达标（需要显示官网pth精度与om模型的精度）
  # npu性能数据(由于msame工具不支持多batch，所以只测试了bs1的性能)
  # 验收结果： 是 / 否
  # 备注： 目标pth精度42.96；bs1验收om精度42.92；精度下降不超过1%；无运行报错，报错日志xx 等
  # 备注： 验收310测试性能bs1:61.538FPS；无运行报错，报错日志xx 等
  
  # 在t4上测试bs1性能
  bash perf.sh
  # 验收结果： OK / Failed
  # 备注： 验收基准测试性能bs1:94.281FPS；无运行报错，报错日志xx 等
  
  # 310性能是否超过基准： 否
  t4:310=(94.281/61.538)1.53倍基准
```

### 6.2 开源精度

BertSum在线训练精度：

42.96%

### 6.3 离线推理精度

42.95%

### 6.3 精度对比

由于源码采用的是动态shape，而离线推理是通过加padding固定住shape进行推理的，所以精度会有损失，因此和同一分辨率下的在线推理进行对比。对比方式：三个尺度求和取平均。

## 7 性能对比

- **310性能数据**
- **T4性能数据**
- **性能对比**

### 7.1 310性能数据

每张图片平均耗时：65.06ms，所以310吞吐率为：1000/65×4=61.538

说明：由于msame不支持多batch，所以此处只测了bs1的性能。

### 7.2 T4性能数据

T4性能为：94.281

### 7.3 性能对比

batch1：94.281>61.538