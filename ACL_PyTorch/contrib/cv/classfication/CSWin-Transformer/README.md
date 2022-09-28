# CSWin-Transformer模型PyTorch离线推理指导

## 1. 模型概述
[论文地址](https://arxiv.org/pdf/2107.00652.pdf)

[代码地址](https://github.com/microsoft/CSWin-Transformer)

## 2. 环境准备 
### 2.1 环境说明
```shell
pip install -r ./requirements.txt
```
### 2.2 环境安装
#### 2.2.1 获取，修改与安装开源模型代码  
```
首先克隆本代码仓
cd CSWin-Transformer
git clone https://github.com/microsoft/CSWin-Transformer.git
cd CSWin-Transformer
git reset f111ae2f771df32006e7afd7916835dd67d4cb9d --hard
cd ..
patch -p0 ./CSWin-Transformer/models/cswin.py diff.patch   (把补丁应用到模型代码上)
cp CSWin_Transformer_preprocess.py ./CSWin-Transformer  (把前处理脚本粘贴到源代码仓中)
cp CSWin_Transformer_postprocess.py ./CSWin-Transformer (把后处理脚本粘贴到源代码仓中)
cp CSWin_Transformer_pth2onnx.py ./CSWin-Transformer    (pth2onnx.py脚本放到源代码仓中)
cd CSWin-Transformer
```



#### 2.2.3 获取权重文件  

[cswin_small_224.pth](https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_small_224.pth)

保证权重文件的位置和pth2onnx.py脚本在同一个文件夹中

#### 2.2.4 获取数据集     
[获取imagNet-1K数据集](https://www.image-net.org/download.php)

#### 2.2.5 获取ais_infer工具
[获取ais_infer工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer) ，按照教程安装好这个工具。

## 3. 数据预处理 
### 3.1 数据集预处理

新建二进制文件目录

```
mkdir ${savePath}
```

- `--${savepath}`：处理后二进制文件存放的目录

将数据集处理为可输入推理工具ais_infer的格式

```shell
python CSWin_Transformer_preprocess.py --data ${dataset_path} --savepath ${savePath}
```
参数说明
- `--data`：数据集位置    路径为验证集文件夹的上一级，比如： /opt/npu/imagenet/
- `--savepath`：输出的二进制文件存放的文件夹位置

## 4. 模型转换
### 4.1 执行pth2onnx脚本生成onnx文件
```shell
python CSWin_Transformer_pth2onnx.py ${batch_size} ${input_path} ${output_model}
```
参数说明
- `参数1`: 转出onnx模型的batch_size
- `参数2`: 输入权重文件的路径，比如   ./cswin_small_224.pth
- `参数3`: 输出onnx模型的路径，注意此处要加上模型名称，比如   ./cswin_bs1.onnx

### 4.2 onnx模型转om模型
#### 4.2.1 设置环境变量
```shell
source ${HOME}/ascend-toolkit/set_env.sh
```
说明
设置CANN的环境变量，默认在/usr/local/Ascend下

#### 4.2.2 使用ATC命令将onnx模型转换为om模型
执行前使用`npu-smi info`查看设备状态，确保device空闲
```shell
export TUNE_BANK_PATH=custom_tune_bank/bs${bs}

atc --model=cswin_bs${bs}.onnx --framework=5 --output=cswin_bs${bs} --input_format=NCHW --input_shape="input:${bs},3,224,224"  --output_type=FP16 --log=error --soc_version=Ascend${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
```
参数说明
- `--model`: 输入的onnx模型路径。
- `--output`:输出的文件名。 
- `--input_format`: 输入形状的格式。
- `--input_shape`: 模型输入的形状。
- `--log`: 设置ATC模型转换过程中日志的级别
- `--soc_version`:目标芯片类型，如Ascend310、Ascend310P，${chip_name}可通过`npu-smi info`指令查看
- `${bs}`：为转出om模型的batch size
- `--optypelist_for_implmode`：设置optype列表中算子的实现方式，这里设为Gelu
- `--op_select_implmode`：选择转换模式为高精度还是高性能，本次使用高性能模式

## 5 ais_infer推理
### 5.1 ais_infer工具简介
ais_infer工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310和Ascend310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程。
### 5.2 设置环境变量
```shell
source ${HOME}/ascend-toolkit/set_env.sh
```
说明
设置CANN的环境变量，其中${HOME}为CANN包安装路径，默认在/usr/local/Ascend下

### 5.3 执行推理
执行时使用`npu-smi info`查看设备状态，确保device空闲
```shell
mkdir ${output_dir}

python ais_infer.py  --model ${model_path}/cswin_bs${bs}.om --input ${input} --output ${output} --outfmt TXT --batchsize ${batch_size}

cd ${output_dir}
rm -f sumary.json
cd ${path_postprocess}
```
参数说明
- `--model`:om模型的路径

- `--input`:预处理后的数据集二进制文件位置

- `--output`:推理结果位置 

- `--outfmt`:推理结果的格式，此处选择TXT

- `${output_dir}`:是最终推理结果存放的具体文件夹，通常以推理开始的时间作为文件夹名称， 因为ais_inder推理工具会生成一个sumary.json文件，把他删掉之后后处理才能正常进行。

- `${model_path}`:是om模型存放的文件夹的路径

- `--batchsize`:模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。用于结果吞吐率计算。

- `${path_postprocess}`:后处理脚本 CSWin_Transformer_postprocess.py 所在的路径，此处绝对路径，比如   /home/Liu/CSWin-Transformer/

  

## 6. 精度和性能对比
### 6.1 离线推理精度
回到代码目录，运行如下脚本评测精度，运行后精度结果保存在result.json文件中
```shell
python CSWin_Transformer_postprocess.py ${output} ${val_label} ./ result.json
```
参数说明
- `参数1`：ais_infer推理结果文件所在路径
- `参数2`：数据集标注标签路径
- `参数3`：imagenet数据集的val_label存放地址
- `参数4`：结果保存的文件
### 6.2 npu性能数据
工具推理后生成result.json文件，其中thuroughput即为310P单卡吞吐率，Interface turoughputRate *4为310单卡吞吐率，运行脚本计算310单卡吞吐量，计算结果回显
## 7. 评测结果：
|          模型          | 官网pth精度  | 310P离线推理精度 | T4基准性能  |    310P性能 |
| :--------------------: | :----------: | :--------------: | :---------: | ----------: |
| CSWin-Transformer bs1  | top1 : 83.6% |  top1 : 83.31%   | 81.0707fps  | 104.0722fps |
| CSWin-Transformer bs16 | top1 : 83.6% |  top1 : 83.26%   | 211.3101fps | 206.3643fps |

