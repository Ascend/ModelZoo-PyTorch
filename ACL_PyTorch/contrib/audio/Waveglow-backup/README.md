## Waveglow 模型离线推理指导

### 一、环境准备

**Ascend环境：** CANN 5.1.rc1

#### 1. 源码下载
```
git clone https://github.com/NVIDIA/waveglow.git
cd waveglow
git submodule init
git submodule update
git apply ../Waveglow.patch
```
#### 2. 创建conda环境
```
conda create --name waveglow python=3.7.5
conda activate waveglow
```
#### 3.安装依赖包
```
pip3 install -r requirements.txt --no-deps
```
### 二、模型转换

#### 1. pt转ONNX模型

```
# 获取pt文件，保存到当前工作目录
wget https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view 

# 执行Waveglow_pth2onnx.py脚本，生成onnx模型文件
python3 ../Waveglow_pth2onnx.py -i ./waveglow_256channels_universal_v5.pt -o ./
```
**参数说明：**
> -i pt文件路径  
> -o onnx文件保存路径

#### 2. onnx转om模型

使用atc工具将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

310P:
```
atc --model=waveglow.onnx \
    --output=waveglow \
    --input_shape="mel:1,80,-1" \
    --framework=5 \
    --input_format=ND \
    --soc_version=Ascend${chip_name} \
    --log=error \
    --dynamic_dims="154;164;443;490;651;699;723;760;832;833"
```
**参数说明：**
> om模型转换耗时较长，请耐心等待  
> 测试集共包含10条数据，所以参数 '--dynamic_dims' 设为10条数据的shape  
> \\${chip\_name}可通过 npu-smi info 指令查看，如下图标注部分
![](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)  


### 三、数据集预处理
#### 1. 获取数据集
下载[LJSpeech-1.1数据集](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2)，解压至当前目录
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar jxvf LJSpeech-1.1.tar.bz2
```
解压后数据集目录结构如下:
```
${data_path}
    |-- LJSpeech-1.1
        |-- wavs
        |    |-- LJ001-0001.wav
        |    |-- LJ001-0002.wav
        |    |-- …
        |    |-- LJ050-0278
        |-- metadata.csv
    |-- README
```
#### 2. 数据前处理
```
# 测试集为LJSpeech-1.1数据集中前10条数据
ls LJSpeech-1.1/wavs/*.wav | head -n10 > test_files.txt
rm -rf ./data
mkdir data
python3 ../Waveglow_preprocess.py -f ./test_files.txt -c ./config.json -o ./data/
```
**参数说明：**
> -f 测试集数据名  
> -c 模型配置json文件  
> -o 前处理结果存放路径
### 四、 离线推理
#### 1. 安装ais_bench推理工具
请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

#### 2. 离线推理
```
# 设置环境变量，请以实际安装环境配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 推理前使用 'npu-smi info' 命令查看 device 是否在运行其它推理任务，确保 device 空闲
npu-smi info

# 执行离线推理
rm -rf result/
mkdir result
sh ../Waveglow_inference.sh  ./data/
```
**参数说明：**
>第一个参数为前处理结果存放路径 


### 五、 后处理
#### 1. 推理结果 
执行Waveglow_postprocess.py脚本对ais_bench推理结果进行后处理，得到'.wav'音频文件
```
rm -rf ./synwavs
mkdir synwavs
python3 ../Waveglow_postprocess.py -f ./result/ -o ./synwavs/
```
**参数说明：**
> -f 推理结果路径  
> -o 后处理结果存放路径  

#### 2. 性能数据
使用ais_bench推理工具获得性能数据：
```
python3 -m ais_bench --model "./waveglow.om" --output "./output/" --outfmt BIN --dymDims mel:1,80,832 --batchsize 1
```
Interface throughput Rate:0.59fps，即是batch1 310P单卡吞吐率  

### 3. 性能对比

性能对比表格如下：
|           |  310P    | T4        |  310P/T4  |
| --------- | -------- | -------   | --------- | 
| bs1       | 0.59 | 0.037  |  15.946   |

最优的310P性能为T4性能的15.946倍。

