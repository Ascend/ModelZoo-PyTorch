## 1、环境准备
### 1.1 下载pth模型
下载[pth模型](https://pan.baidu.com/s/1xGZFEQTjhsP4sj0Lvf_sXg)到推理目录下
提取码: e5ci 

### 1.2 下载onn_tools用于模型优化
下载[onnx_tools](https://gitee.com/zheng-wengang1/onnx_tools/tree/master/OXInterface)到推理目录下

### 1.2 安装必要环境
```
pip install -r requirements.txt  
```

### 1.3 获取,安装,修改开源模型代码
```
git clone https://github.com/HRNet/HRNet-Semantic-Segmentation.git 
cd HRNet-Semantic-Segmentation
git reset 0bbb2880446ddff2d78f8dd7e8c4c610151d5a51 --hard
patch -p1 < ../HRNet.patch 
cd ..
```

### 1.4 下载数据集
下载[cityscpaes数据集](https://www.cityscapes-dataset.com)
解压后数据集目录结构如下:
```c
  ${data_path}
  |-- cityscapes
      |-- gtFine
      |    |-- test
      |    |-- train
      |    |-- val
      |-- leftImg8bit
           |-- test
           |-- train
           |-- val 
```

### 1.5 获取benchmark工具 
参考[CANN5.0.1 推理benchmark工具用户指南01](https://support.huawei.com/enterprise/zh/doc/EDOC1100191895?idPath=23710424%7C251366513%7C22892968%7C251168373)下载benchmark工具，并放置在推理目录下


## 2、离线推理
310上执行`npu-smi info`查看设备状态，确保device空闲      
### 2.1 模型转换
执行如下指令会生成hrnet.onnx, hrnet_bs4.om, hrnet_bs1.om
```
bash test/pth2om.sh
```
### 2.2 精度统计及性能
执行如下指令会在result目录下生成模型的推理结果文件以及推理性能文件,这里已经将数据集放置在/opt/npu/目录下
```
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/
```
推理性能文件参考如下:
```
[e2e] throughputRate: 135.162, latency: 73985.3
[data read] throughputRate: 138.155, moduleLatency: 7.23827
[preprocess] throughputRate: 137.422, moduleLatency: 7.27687
[infer] throughputRate: 138.1, Interface throughputRate: 554.397, moduleLatency: 4.00654
[post] throughputRate: 138.098, moduleLatency: 7.24122
```
同时打印精度测试结果并保存在result_bs.json文件中:
```
IoU_array: [0.98405628 0.86770446 0.93294902 0.54741631 0.63803344 0.70017193
 0.74378167 0.82155113 0.92945463 0.63801228 0.9534655  0.84110389
 0.65776715 0.9574613  0.8731766  0.91712681 0.84788631 0.69007423
 0.79812632]
 mean_IoU: 0.8073325927379879
```
将onnx模型置于t4设备，获取onnx模型在gpu上推理的性能信息
```
bash test/perf_g.sh
```

### 2.3 评测结果

|3D HRNet模型|  gpu吞吐率| 310吞吐率 |  目标精度| 310精度|
|--|--|--|--|--|
|  bs1|  5.85fps| 4.90fps|81.6%|80.85%|
|  bs4| 5.75fps| 4.78fps|81.6% |80.85%|
