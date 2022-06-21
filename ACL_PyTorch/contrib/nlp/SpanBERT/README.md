- [1. 模型概述](#1)
    - [论文地址](#11)
    - [代码地址](#12)
- [2. 环境说明](#2)
    - [深度学习框架](#21)
    - [python第三方库](#22)
- [3. 模型转换](#3)
    - [pth转onnx模型](#31)
    - [onnx转om模型](#32)
- [4. 数据预处理](#4)
    - [数据处理](#41)
- [5. 离线推理](#5)
    - [msame工具概述](#51)
    - [离线推理](#52)
    - [精度和性能比较](#53)

## <a name="1">1. 模型概述</a>
### <a name="11">1.1 论文地址</a>
[spanBert论文]([https://arxiv.org/pdf/1907.10529.pdf])

### <a name="12">1.2 代码地址</a>
[spanBert代码](https://github.com/facebookresearch/SpanBERT)

## <a name="2">2. 环境说明</a>
### <a name="21">2.1 深度学习框架与第三方库</a>
```
CANN 5.1.RC1
boto3==1.24.3
requests==2.27.1
tqdm==4.64.0
sympy==1.10.1
decorator==5.1.1
numpy==1.21.6
torch==1.8.1
onnxruntime==1.11.1
```

> **说明：**  
> pytorch，torchvision和onnx:(X86架构)可以通过pip方式安装或官方下载whl包安装; (Arm架构)可以通过源码编译安装   
> 其他第三方库: 可以通过 pip3 install -r requirements.txt 进行安装

### <a name="21">2.2 权重文件及原仓代码数据集下载</a>

由于后续需要用到部分源仓代码，需要使用以下命令进行下载

```bash
git clone https://github.com/facebookresearch/SpanBERT.git
```

使用源仓提供的bin以及config文件，可使用download_finetuned.sh进行下载,第一个参数为权重下载后存放的文件夹，第二个参数为任务名

```bash
bash SpanBERT/code/download_finetuned.sh model_dir squad1
```

## <a name="3">3. 模型转换</a>

一步式从pth权重文件转om模型的脚本，能够由bin权重文件以及config文件生成动态Batch的onnx模型和指定bacth_size的om模型，以下代码batch_size设为1：
```bash
bash ./test/pth2om.sh --batch_size=1 --config_file="./model_dir/squad1/config.json" --checkpoint="./model_dir/squad1/pytorch_model.bin" --chip_name="310P3"
```
运行后会生成如下文件：
```bash
├── spanBert_dynamicbs.onnx
├── spanBert_bs1.om
```

### <a name="31">3.1 pth转onnx模型</a>
1. 设置环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 权重文件: 

  通过download_finetuned.sh下载，model_dir若要更改的话，后面的脚本也需要将出现model_dir的位置替换成更改后的文件夹名称

```bash
bash download_finetuned.sh model_dir squad1
```

3. 执行spanBert_pth2onnx.py脚本，生成onnx模型文件 
```bash
python spanBert_pth2onnx.py  \
--config_file ./model_dir/squad1/config.json  \
--checkpoint ./model_dir/squad1/pytorch_model.bin
```
其中"config_file"表示模型config文件的路径，"checkpoint"表示模型bin文件的路径，执行后会在当前路径生成spanBert_dynamicbs.onnx  

### <a name="32">3.2 onnx转om模型</a>
1. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha002/infacldevg/atctool)

   chip_name可以通过npu-smi info指令查看，运行指令后显示的Name_device即为要输入的参数:

```bash
atc --framework=5 --model=./spanBert_dynamicbs.onnx --output=./spanBert_bs1 --input_format=ND --input_shape="input_ids:1,512;token_type_ids:1,512;attention_mask:1,512" --log=error --soc_version=Ascend${chip_name}
```

此处为batch_size为1的情况，若batch_size不为1，则需要将"output"中的bs1替换为新的batch_size，通过"input_shape"中的1也需要替换为batch_size

## <a name="4">4. 数据预处理</a>

数据预处理过程包含在 test/eval_acc_perf.sh 的脚本中，可以直接运行，完成预处理以及推理：

```bash
bash test/eval_acc_perf.sh --batch_size=1 --datasets_path="/opt/npu/squad1"
```

### <a name="41">4.1 数据处理</a>

1. 设置环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 执行输入数据的生成脚本，生成模型输入的bin文件

```bash
python spanBert_preprocess.py \
    --dev_file /opt/npu/squad1/dev-v1.1.json \
    --batch_size 1
```
其中"dev_file"表示处理前原数据集的路径，"batch-size"表示生成数据集对应的batch_size，本项目使用的推理工具为msame，需要针对不同的batch_size生成不同的输入数据，且多输入时不同文件夹的bin需要保持名称一致
运行后，将会得到如下形式的文件夹：

```
├── input_ids 
│    ├──0.bin
│    ├──......     	 
├── input_mask 
│    ├──0.bin
│    ├──...... 
├── segment_ids 
│    ├──0.bin
│    ├──......
```

## <a name="5">5. 离线推理</a>
执行一步式推理前，请先准备msame离线推理工具  
一步式进行输入数据的准备，模型离线推理和NPU性能数据的获取：

```bash
bash ./test/eval_acc_perf.sh --batch_size=1 --datasets_path=/opt/npu/squad1
```
运行后会生成如下文件/文件夹：
```bash
├── input_ids 
│    ├──0.bin
│    ├──......     	 
├── input_mask 
│    ├──0.bin
│    ├──...... 
├── segment_ids 
│    ├──0.bin
│    ├──......
├── msame_bs1.txt         # msame推理过程的输出
├── result            
│    ├── outputs_bs1_om   # 模型的输出(文件夹)
```

### <a name="51">5.1 msame工具概述</a>
msame模型推理工具，其输入是om模型以及模型所需要的输入bin文件，其输出是模型根据相应输入产生的输出文件。获取工具及使用方法可以参考[msame模型推理工具指南](https://gitee.com/ascend/tools/tree/master/msame)
### <a name="52">5.2 离线推理</a>
1. 设置环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 执行离线推理
运行如下命令进行离线推理：
```bash
./msame --model "./spanBert_bs1.om" \
        --input "./input_ids,./segment_ids,./input_mask" \
        --output "./result/outputs_bs1_om" \
        --outfmt BIN > ./msame_bs1.txt
```
模型输出格式是BIN，输出保存在"output"参数指定的文件夹中，同时会生成推理的日志文件msame_bs1.txt

### <a name="53">5.3 精度和性能比较</a>
1. 性能数据的获取
通过给test/parser.py指定推理后的日志文件，可以得到离线推理的性能数据
```bash
python3 test/parse.py --result-file ./msame_bs1.txt --batch-size 1
```
其中"result-file"表示性能数据的地址和名称，"batch-size"表示性能测试时模型对应的batch size

2. 精度数据的计算
精度计算利用spanBert_postprocess.py脚本
```bash
python spanBert_postprocess.py \
	--do_eval \
	--model spanbert-base-cased \
	--dev_file /opt/npu/squad1/dev-v1.1.json \
	--max_seq_length 512 \
	--doc_stride 128 \
	--eval_metric f1 \
	--fp16 \
	--bin_dir ./result/outputs_bs1_om \
    --eval_batch_size 1
```
其中"bin_dir"表示离线推理输出所在的文件夹，"dev_file"表示squad1验证集路径，"batch-size"表示精度测试时模型对应的batch size

3. 精度数据的获取
通过给test/parser.py指定推理后的日志文件，可以得到离线推理的性能数据
```bash
python3 test/parse.py --result-file ./result/result_bs1.json
```
| 模型      | 参考精度  | 310P精度  |
| :------: | :------: | :------: |
| SpanBERT bs1 | f1:92.4% | f1:93.95381607527929% |

|     模型      |       性能基准        |     310P性能     |
| :-----------: | :-------------------: | :--------------: |
| SpanBERT bs1  | 10.043299911570772fps |    45.455fps     |
| SpanBERT bs4  | 10.60900682143337fps  |    27.791fps     |
| SpanBERT bs8  | 10.464110083241215fps |    34.646fps     |
| SpanBERT bs16 | 10.823289177296912fps | 34.823109486fps  |
| SpanBERT bs32 | 10.029470268741672fps | 35.2621354088fps |
| SpanBERT bs64 | 9.992470491681907fps  | 35.4670848827fps |

