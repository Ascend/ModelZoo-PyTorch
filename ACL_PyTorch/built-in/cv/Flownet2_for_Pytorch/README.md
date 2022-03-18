# Flownet2模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt
```

2.获取，修改与安装开源模型代码

```
git clone https://github.com/NVIDIA/flownet2-pytorch.git
cd flownet2-pytorch && git checkout 2e9e010c98931bc7cef3eb063b195f1e0ab470ba
patch -p1 < ../flownet2.patch
cd ..
# 安装onnx改图工具
git clone https://gitee.com/Ronnie_zheng/MagicONNX
cd MagicONNX && git checkout 8d62ae9dde478f35bece4b3d04eef573448411c9
pip install .
cd ..
```

3.获取权重文件

获取FlowNet2_checkpoint.pth.tar，放置到当前工作目录

4.数据集

获取MPI-Sintel-complete数据集，解压到当前目录


5.[获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)  
将msame放到当前目录

## 2 离线推理

1. 模型转换

```
# convert onnx
python3.7 pth2onnx.py --batch_size 1 --input_path ./FlowNet2_checkpoint.pth.tar --out_path ./models/flownet2_bs1.onnx --batch_size 1
# optimize onnx
python3.7 -m onnxsim ./models/flownet2_bs1.onnx ./models/flownet2_bs1_sim.onnx
python3.7 fix_onnx.py ./models/flownet2_bs1_sim.onnx ./models/flownet2_bs1_sim_fix.onnx
# 310需要采用混合精度，否则有精度问题；710上采用FP16精度正常
atc --framework=5 --model=models/flownet2_bs1_sim_fix.onnx --output=models/flownet2_bs1_sim_fix --input_format=NCHW --input_shape="x1:1,3,448,1024;x2:1,3,448,1024" --log=debug --soc_version=Ascend310 --precision_mode=allow_mix_precision
atc --framework=5 --model=models/flownet2_bs1_sim_fix.onnx --output=models/flownet2_bs1_sim_fix_710 --input_format=NCHW --input_shape="x1:1,3,448,1024;x2:1,3,448,1024" --log=debug --soc_version=Ascend710
```

2. 数据预处理

```
python3.7 preprocess.py --batch_size 1 --dataset ./MPI-Sintel-complete/training --output ./data_preprocessed_bs1
```

3. 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  

```
./msame --model models/flownet2_bs1_sim_fix.om --input data_preprocessed_bs1/image1/,data_preprocessed_bs1/image2/ --output output_bs1/
```

4. 数据后处理

```
python3.7 evaluate.py --gt_path ./data_preprocessed_bs1/gt --output_path ./output_bs1/ --batch_size 1
```

 **评测结果：**

 bs16占据内存过大，无法导出

|     模型     |      pth精度       |  310离线推理精度   |   710离线推理精度   | 基准性能  | 310性能 | 710性能  |
| :----------: | :----------------: | :----------------: | :-----------------: | :-------: | :-----: | :------: |
| flownet2 bs1 | Average EPE: 2.150 | Average EPE: 2.184 | Average EPE: 2.1578 | 11.65fps | 3.07fps | 16.81fps |