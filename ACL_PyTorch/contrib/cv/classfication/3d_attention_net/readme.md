# 一. 环境准备
1. 安装必要的依赖，运行所需的依赖详见requirements.txt
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

4. 获取benchmark工具
	将benchmark.x86_64放到当前目录。
5. 获取OXInterface.py
```
git clone https://gitee.com/zheng-wengang1/onnx_tools.git
cd onnx_tools
git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
cd ..
```

# 二. 数据预处理
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

5. 生成预处理后的数据集信息文件
```
python3.7 gen_dataset_info.py bin ./pre_process_result/ ./3d_attention_net_prep_bin.info 32 32
```

# 三. pkl文件转om模型
1. 读取源代码仓中的pkl文件，将原始模型转换为onnx
```
python3.7 3d_attention_net_pkl2onnx.py
```

2. 对onnx模型中的resize操作进行优化
```
python3.7 resize_optimize.py
```

3. 通过Autotune进行性能调优并转换为om模型
```
bash test/3d_attention_net_onnx2om.bash
```

# 四. om模型离线推理，性能及精度测试 
1. bs1离线推理测试
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=3d_attention_net_resize_autotune_optimized_bs1.om -input_text_path=3d_attention_net_prep_bin.info -input_width=32 -input_height=32 -output_binary=False -useDvpp=False
python3.7 3d_attention_net_postprocess.py 0
```
python3.7 3d_attention_net_postprocess.py $DEVICE_ID  
传入的第一个参数DEVICE_ID为指定device的输出，应与benchmark传入的device_id保持一致，下同
	
2. bs16离线推理测试
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=3d_attention_net_resize_autotune_optimized_bs16.om -input_text_path=3d_attention_net_prep_bin.info -input_width=32 -input_height=32 -output_binary=False -useDvpp=False
python3.7 3d_attention_net_postprocess.py 0
```

3. GPU性能测试
```
bash perf_g.sh
```

4. 性能&精度对比


|模型|官网pkl精度|310离线推理精度|基准性能|310性能|精度对比<br>(310/基准)|性能对比<br>(310/基准)|
|-|:-:|:-:|:-:|:-:|:-:|:-:|
|3d_attention_net_bs1|Top-1：95.4%|Top-1：95.34%|659.2fps|1479.5fps|99.94%|1479.5/659.2|
|3d_attention_net_bs16|Top-1：95.4%|Top-1：95.34%|3494.16fps|3980.4fps|99.94%|3980.4/3494.16|

