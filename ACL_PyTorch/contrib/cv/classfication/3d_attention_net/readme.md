# 3D_A_Net模型PyTorch离线推理指导

## 一. 环境准备

1. 创建环境安装必要的依赖，运行所需的依赖详见requirements.txt

```
conda create -n yxy python=3.7
pip install -r requirements.txt
```

2. 获取开源模型代码及权重文件

```
git clone https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch.git
cd ResidualAttentionNetwork-pytorch
git checkout 44d09fe9afc0d5fba6f3f63b8375069ae9d54a56
cd Residual-Attention-Network
cp -r model ../..
cp model_92_sgd.pkl ../..
cd ../..
```

3. 由于python版本问题，原模型代码在执行过程中会出现数据类型转换问题，依次执行以下命令

```
cd model/
patch -p1 <../3d_attention_net.patch
cd ..
```

4. 获取推理工具

```
git clone https://gitee.com/ascend/tools.git
cd ./tools/ais-bench_workload/tool/ais_infer/
pip3 wheel ./backend/ -v
pip3 wheel ./ -v

pip3 install ./aclruntime-{version}-cp37-cp37m-linux_xxx.whl 
pip3 install ./ais_bench-{version}-py3-none-any.whl
source  /usr/local/Ascend/ascend-toolkit/set_env.sh
```

5. 获取OXInterface.py

```
git clone https://gitee.com/zheng-wengang1/onnx_tools.git
cd onnx_tools
git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
cd ..
```

## 二. 数据预处理

1. 获取CIFAR-10数据集

```
website：https://www.cs.toronto.edu/~kriz/cifar.html 
#Version：CIFAR-10 python version
#md5sum：c58f30108f718f92721af3b95e74349a
```

2. 上传数据集

```
mkdir data
```

3. 将下载的CIFAR-10数据集上传至data文件夹，而后执行如下命令：

```
tar -zxvf data/cifar-10-python.tar.gz -C data/
```

4. 数据预处理

```
python3.7 3d_attention_net_preprocess.py
```

## 三. pkl文件转om模型

1. 读取源代码仓中的pkl文件，将原始模型转换为onnx

```
python3.7 3d_attention_net_pkl2onnx.py
```

2. 对onnx模型中的resize操作进行优化
```
python3.7 resize_optimize.py
```

3. 通过ATC将onnx模型转换为om模型（以bs1、bs4为例），该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=3d_attention_net_resize_optimized.onnx --output=3d_attention_net_resize_autotune_optimized_bs1 --input_format=NCHW --input_shape="image:1,3,32,32" --log=debug -soc_version=Ascend${chip_name}

atc --framework=5 --model=3d_attention_net_resize_optimized.onnx --output=3d_attention_net_resize_autotune_optimized_bs4 --input_format=NCHW --input_shape="image:4,3,32,32" --log=debug -soc_version=Ascend${chip_name}
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

## 四. om模型离线推理，性能及精度测试

1. 执行推理（以bs1、bs4为例）

```
python3.7 -m ais_bench --model ./3d_attention_net_resize_autotune_optimized_bs1.om --input "./pre_process_result" --output ./lmcout --outfmt TXT --batchsize 1
python3.7 -m ais_bench --model ./3d_attention_net_resize_autotune_optimized_bs4.om --input "./pre_process_result" --output ./lmcout --outfmt TXT --batchsize 4
```

- 参数说明：
  - --model：模型地址。
  - --input：预处理完的数据集文件夹。
  - --output：推理结果保存地址。
  - --outfmt：推理结果保存格式。
  - --batchsize：bs。

2. 精度验证

```
python3.7 3d_attention_net_postprocess.py --pred_res_path='.lmcout/xxx'
```
- 参数说明：
  - --pred_res_path：推理结果地址。
  - --新建result文件夹，用于保存精度数据。

3. GPU性能测试（T4，以bs1、bs4为例）

```
trtexec --onnx=3d_attention_net.onnx --fp16 --shapes=image:1x3x32x32 --workspace=5000 --threads
trtexec --onnx=3d_attention_net.onnx --fp16 --shapes=image:4x3x32x32 --workspace=5000 --threads
```

4. 性能&精度对比

|       Input         |310性能|310P性能 |T4性能|310P/310|310P/T4|
| :-----------------: | :--:  | :---: | :--: | :-:| :-: |
|3d_attention_net_bs1 |672.122|1017.57| 357.85|1.51| 2.84|
|3d_attention_net_bs4 |2845.32|3572.93|2109.08|1.26| 1.69|
|3d_attention_net_bs8 |3317.80|4798.82|2938.48|1.45|1.633|
|3d_attention_net_bs16|3533.76|7806.96|3373.12|2.21|2.314|
|3d_attention_net_bs32|3633.64|7742.07|3663.36|2.13|2.113|
|3d_attention_net_bs64|3700.00|5927.82|3930.88|1.60| 1.50|
|最优batch             |3700.00|7806.96|3930.88|2.11|1.986|
