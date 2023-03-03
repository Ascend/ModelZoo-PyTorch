​       
# UNet模型PyTorch离线推理指导

## 1 环境准备 

### 1.1 安装必要的依赖

测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3 install -r requirements.txt  
```
说明：PyTorch选用开源1.7.0版本

requirements.txt中的相关依赖如下：

```
torch==1.7.0
torchvision==0.8.0
onnx==1.8.1
opencv-python==4.5.1.48
onnx-simplifier==0.3.3
Pillow==8.1.2
protobuf==3.20.0
decorator
tqdm
```

### 1.2 获取，修改与安装开源模型代码  

获取[ATC UNet (FP16)](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/02704892d4914bb191b5b11c86e7c94c)源码包
单击“立即下载”，上传源码包到服务器任意目录并解压

```
├── preprocess_unet_pth.py           //数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件
├── ReadMe.md
├── UNet.pth                         //训练后的权重文件
├── UNet_bs1.om                      //batchsize为1的离线模型
├── UNet_dynamic_bs.onnx             //onnx格式的模型文件
├── UNet_atc.sh                      //onnx模型转换om模型脚本
├── unet_pth2onnx.py                 //用于转换pth模型文件到onnx模型文件
├── revise_UNet.py                   //用于删除多余的pad节点，提升性能
└── postprocess_unet_pth.py          //验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy
```

下载代码仓并做预处理

```shell
git clone https://github.com/milesial/Pytorch-UNet.git
cd Pytorch-UNet
git reset --hard 6aa14cb
mv ./Pytorch-UNet ./Pytorch_UNet
```

### 1.3  安装ais_bench推理工具

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

## 2准备数据集   

###   2.1 获取原始数据集

本模型支持carvana数据集，用户需自行获取数据集。数据集train.zip以及train_masks.zip分别作为训练和标签文件。上传并解压到ModleZoo源码包的根目录中。

### 2.2 数据预处理

将原始数据（.jpg）转化为二进制文件（.bin）。执行preprocess_unet_pth.py脚本。

```
python3 preprocess_unet_pth.py /opt/npu/carvana/train ./prep_bin
```

第一个参数为原始数据验证集所在路径，第二个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。运行成成功后，生成二进制文件夹prep_bin。

## 3 离线推理 

### 3.1 模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

#### 3.1.1 获取权重文件。

[UNet.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Unet/PTH/UNet.pth)

#### 3.1.2 导出onnx文件

将源码包中的unet_pth2onnx.py脚本移到Pytorch_UNet目录，并使用unet_pth2onnx.py脚本将.pth文件转换为.onnx文件

```
mv unet_pth2onnx.py ./Pytorch_UNet/
python3 ./Pytorch_UNet/unet_pth2onnx.py ./UNet.pth ./UNet_dynamic_bs.onnx
```

第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。运行成功后，在当前目录生成UNet_dynamic_bs.onnx模型文件。


使用onnxsim精简onnx文件（需安装onnx-simplifer）

```
python3 -m onnxsim --input-shape="1,3,572,572" UNet_dynamic_bs.onnx unet_carvana_sim.onnx
```

运行成功后，在当前目录生成unet_carvana_sim.onnx模型文件。


使用revise_UNet.py删除精简后的onnx文件中的多余pad节点。若用户执行自己的模型需要在脚本中修改输入和输出的模型名称。该步骤去除多余的pad节点，用于提高模型性能，用户可根据shape大小自行决定是否去除pad节点。

```
python3 revise_UNet.py
```

运行成功后，在当前目录生成unet_carvana_sim_final.onnx文件用于转om模型。

使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。

#### 3.1.3 使用ATC工具将ONNX模型转OM模型

##### 3.1.3.1配置环境变量

source /usr/local/Ascend/ascend-toolkit/set_env.sh

此环境变量可能需要根据实际CANN安装路径修改

##### 3.1.3.2执行命令，将.onnx文件转为离线推理模型文件.om文件

${chip_name}可通过`npu-smi info`指令查看，例：310P3

```
atc --model=./unet_carvana_sim_final.onnx --framework=5 --output=UNet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,572,572" --log=info --soc_version=${chip_name}
```

- 参数说明：
  - --model：为ONNX模型文件。
  - --framework：5代表ONNX模型。
  - --output：输出的OM模型。
  - --input_format：输入数据的格式。
  - --input_shape：输入数据的shape。
  - --log：日志级别。
  - --soc_version：处理器型号。

运行成功后生成的UNet_bs1.om文件用于图片输入推理的模型文件。

### 3.2 开始推理验证
安装ais_bench推理工具  

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
#### 3.2.1 使用ais_bench推理工具执行推理

```
python3 -m ais_bench --model ./UNet_bs1.om --input "./prep_bin" --batchsize 1 --output new_result/
```

输出文件夹通常根据系统当前时间命名，如2022_08_05-13_01_09，为便于操作，可以更改输出文件夹的名字

```
cd new_result
mv 2022_08_05-13_01_09/ bs1
cd ..
```

#### 3.2.2 数据处理

处理summary.json文件，依据json文件信息更改推理输出文件的名字（注意根据实际需要更改json_parser.py文件中推理输出对应的路径）

```
python3 json_parser.py "new_result/bs1/"
```

#### 3.2.3 精度验证

调用postprocess_unet_pth.py脚本与train_masks标签数据比对，可以获得Accuracy数据。

```
python3 postprocess_unet_pth.py new_result/bs1 /opt/npu/carvana/train/train_masks ./result.txt
```

第一个参数为生成推理结果所在路径，第二个参数为标签数据图片文件夹，第三个参数为保存各个文件IOU计算结果。

**性能验证：** 

|      | 310     | 310P     | T4      | 310P/310 | 310P/T4 |
|------|---------|---------|---------|---------|--------|
| bs1  | 44.3952 | 78.6449 | 38.4451 | 1.7715  | 2.0456 |
| bs4  | 42.5848 | 74.7858 | 35.5334 | 1.7562  | 2.1047 |
| bs8  | 42.3648 | 73.2978 | 35.3507 | 1.7302  | 2.0734 |
| bs16 | 41.9352 | 72.7301 | 35.7969 | 1.7343  | 2.0317 |
| bs32 | 41.8216 | 71.5737 | 35.6670 | 1.7114  | 2.0067 |
| 最优bs | 44.3952 | 78.6449 | 38.4451 | 1.7715  | 2.0456 |

在最优batch下。310P推理性能满足 310P＞1.2倍310 310P＞1.6倍T4。
   

**精度验证：** 

|      | 310      | 310P      |
|------|----------|----------|
| bs1  | 0.986427 | 0.986395 |
| bs16 | 0.986427 | 0.986395 |
