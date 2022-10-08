
# RotatE模型PyTorch离线推理指导

RotatE推理部分实现

### 获取源码
1. 单击“立即下载”，下载源码包。
2. 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。
3. 获取开源代码仓（包含原始数据集）。
```
git clone https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding -b master
cd KnowledgeGraphEmbedding
git reset --hard 2e440e0f9c687314d5ff67ead68ce985dc446e3a
cd ..
```
### 配置环境

1. 创建自己的conda虚拟环境，python版本为3.7.5
```
conda create -n lcyenv python=3.7.5
```
2. 安装所依赖的包
```
pip install numpy
pip install torchvision==0.9.0
pip install onnx==1.9.0
pip install scikit-learn==0.20.0
pip install tqdm
pip install decorator
pip install sympy
```
3. 配置环境变量


### 准备数据集
1. 获取原始数据集，开源代码仓自带数据集。
（六个数据集countries_S1 、countries_S2 、countries_S3 、FB15k、FB15k-237 、wn18 、wn18rr 、YAGO3-10）
2. 数据预处理，batchsize设置为1。
```
python3.7 rotate_preprocess.py --test_batch_size=1 --output_path='bin-bs1/'
```
- 参数说明：
	- --test_batch_size：输入参数。
	- --output_path：输出文件路径。
使用的数据集为FB15k-237，运行后生成“bin-bs1/”文件夹。支持后续不同bs的推理测试。

### 准备推理工具 
1. 下载推理工具-ais_infer
```
git clone https://gitee.com/ascend/tools.git
```
2. 编译、安装推理工具
```
cd ./tools/ais-bench_workload/tool/ais_infer/backend/
pip3.7 wheel ./ #编译 要根据自己的python版本
ls
pip install aclruntime-0.0.1-cp37-cp37m-linux_x86_64.whl
```
### 模型推理
1. 模型转换
使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

**a. 获取权重文件**
```
从源码包中获取权重文件：“checkpoint”。
```

**b. 导出onnx文件**
生成onnx：
```
python3.7 rotate_pth2onnx.py --pth_path="./checkpoint" --onnx_path="./kge_onnx_head.onnx" --mode="head-batch"
python3.7 rotate_pth2onnx.py --pth_path="./checkpoint" --onnx_path="./kge_onnx_tail.onnx" --mode="tail-batch"
```
- 参数说明：
	- --pth_path：checkpoint文件的路径。
	- --onnx_path：生成的onnx的存放路径。
	- --mode：生成head-batch或tail-batch的onnx文件。
获得“kge_onnx_head.onnx”和“kge_onnx_tail.onnx”文件。
- **注意：**
因为RotatE模型是KnowledgeEmbedding类型，模型内部分为head-batch和tail-batch两部分，所以在离线推理的时候，也将模型分为了head-batch和tail-batch两部分。
使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。


**c. 使用ATC工具将ONNX模型转OM模型**（以bs1、bs4为例）
```
atc --framework=5 --model=kge_onnx_head.onnx --output=kge_1_head --input_format=ND --input_shape="pos:1,3;neg:1,14541" --log=error --soc_version=Ascend${chip_name}
atc --framework=5 --model=kge_onnx_tail.onnx --output=kge_1_tail --input_format=NCHW --input_shape="pos:1,3;neg:1,14541" --log=error --soc_version=Ascend${chip_name}

atc --framework=5 --model=kge_onnx_head.onnx --output=kge_4_head --input_format=ND --input_shape="pos:4,3;neg:4,14541" --log=error --soc_version=Ascend${chip_name}
atc --framework=5 --model=kge_onnx_tail.onnx --output=kge_4_tail --input_format=NCHW --input_shape="pos:4,3;neg:4,14541" --log=error --soc_version=Ascend${chip_name}
```
- 参数说明：
	- --model：为ONNX模型文件。
	- --framework：5代表ONNX模型。
	- --output：输出的OM模型。
	- --input_format：输入数据的格式。
	- --input_shape：输入数据的shape。具体的batchsize在input_shape处修改。
	- --log：日志级别。
	- --soc_version：处理器型号。
	- --${chip_name}可通过`npu-smi info`指令查看
运行成功后生成“kge_1_tail.om”和“kge_1_head.om”模型文件。

2. 开始推理验证--使用ais_infer推理工具


**a. 执行推理**（以bs1、bs4为例）
首先，创建保存推理结果的文件夹位置：
```
mkdir -p RotaEout/bs1
```
执行推理：
```
python3.7 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./kge_1_head.om --input "./bin-bs1/head/pos,./bin-bs1/head/neg" --output ./RotaEout/bs1 --outfmt NPY --batchsize 4
python3.7 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./kge_1_tail.om --input "./bin-bs1/tail/pos,./bin-bs1/tail/neg" --output ./RotaEout/bs1 --outfmt NPY --batchsize 4
  
python3.7 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./kge_4_head.om --input "./bin-bs1/head/pos,./bin-bs1/head/neg" --output ./RotaEout/bs4 --outfmt NPY --batchsize 4
python3.7 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./kge_4_tail.om --input "./bin-bs1/tail/pos,./bin-bs1/tail/neg" --output ./RotaEout/bs4 --outfmt NPY --batchsize 4
```
- 参数说明：
	- --model：模型地址
	- --input：预处理完的数据集文件夹
	- --output：推理结果保存地址
	- --outfmt：推理结果保存格式
生成的推理结果保存在RotaEout/bs1文件夹下

**b. 精度验证**（以bs1、bs4为例）
首先删除两个输出文件内的summary.json
```
rm RotaEout/bs32/2022_xx_xx-xx_xx_xx/sumary.json
rm RotaEout/bs32/2022_xx_xx-xx_xx_xx/sumary.json
```
```
python3.7 rotate_postprocess.py --result_path='./RotaEout/bs1' --data_head='./bin-bs1/head' --data_tail='./bin-bs1/tail' > result_bs1.json
python3.7 rotate_postprocess.py --result_path='./RotaEout/bs4' --data_head='./bin-bs1/head' --data_tail='./bin-bs1/tail' > result_bs4.json
```
- 参数说明：
	- --result_path：推理结果对应的文件夹
	- --data_head：处理后的原始数据集--head
	- --data_tail：处理后的原始数据集--tail
生成的精度结果在result_bs1.json文件中


## 推理迁移结果
- 精度结果：

|Model|batch size|310精度|310P精度|
| :------: | :------:  | :------: | :------: | 
|RotaE|1|MRR:0.33555|MRR:0.33568|
|RotaE|4|MRR:0.33555|MRR:0.33568|
|RotaE|8|MRR:0.33555|MRR:0.33568|
|RotaE|16|MRR:0.33555|MRR:0.33568|
|RotaE|32|MRR:0.33555|MRR:0.33568|

精度达标

- 性能测试输入
- 性能结果：（性能均由fps表示）

|Input|310性能|310P性能|aoe后310P性能|T4性能|310P/310|310P/T4|310P/310(aoe后)|310P/T4(aoe后)|
| :------: | :------:  | :------: | :------: |  :------: | :------:  | :------: | :------: | :------: | 
|bs1-head|102.3716|30.9616|**135.0584**|22.5347|0.3024|1.3739|**1.3192**|**5.9933**|
|bs1-tail|102.5444|30.9691|**134.6402**|22.2860|0.3020|1.3896|**1.3129**|**6.0414**|
|bs4-head|104.5953|108.9093|**118.4027**|90.5493|1.0412|1.2027|**1.1320**|1.3076|
|bs4-tail|104.1824|**108.9108**|-|90.6601|**1.0453**|**1.2013**|-|-|
|bs8-head|104.1735|**141.6446**|-|182.8141|**1.3596**|**0.7748**|-|-|
|bs8-tail|104.1767|**141.6109**|-|176.9140|**1.3593**|**0.8004**|-|-|
|bs16-head|126.3315|**141.9402**|-|362.0319|**1.1235**|**0.3920**|-|-|
|bs16-tail|126.1626|**142.0148**|-|176.0865|**1.1256**|**0.7929**|-|-|
|bs32-head|126.5398|**141.3925**|-|716.3820|**1.1173**|**0.1973**|-|-|
|bs32-tail|127.0150|**141.4077**|-|722.9187|**1.1133**|**0.1956**|-|-|

310P的平均精度大于310的1.2倍；310P大于T4的1.6倍。性能达标

