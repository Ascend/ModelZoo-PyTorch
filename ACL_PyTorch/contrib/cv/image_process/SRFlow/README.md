# SRFlow模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码  

```
git clone https://github.com/andreas128/SRFlow -b master 
cd SRFlow 
git reset 8d91d81a2aec17e7739c5822f3a5462c908744f8 --hard
patch -p1 < ../srflow.diff
```

3.获取权重文件  

```
wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow/pretrained_models.zip
unzip pretrained_models.zip
rm pretrained_models.zip
```

4.获取数据集   

```
wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip
unzip datasets.zip
rm datasets.zip
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64放到当前目录下



完成以上步骤后当前目录有以下结构：

```
.
├── srflow.diff
├── benchmark.x86_64
├── env.sh
├── get_info.py
├── LICENSE
├── README.md
├── requirements.txt
├── fusion_switch.cfg
├── SRFlow
│   ├── code
│   ├── datasets
│   │   ├── div2k-validation-modcrop8-gt
│   │   └── div2k-validation-modcrop8-x8
│   ├── LICENSE
│   ├── LICENSES
│   ├── pretrained_models
├── srflow_postprocess.py
├── srflow_preprocess.py
├── srflow_pth2onnx.py
└── test
```



## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  

```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=./SRFlow/datasets 
```
 **评测结果：**   
| 模型      | 官网pth精度  | 310离线推理精度  | gpu性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  |
| SRFlow bs1 | psnr:23.05 | psnr:23.042 | 0.6173fps | 0.5145fps |


备注：

- 需要使用onnxsim
- onnx不支持动态batch
- 若内存充足可导出多batch的onnx模型
- torch1.5.0导出onnx时出现内存不足的问题，torch1.8.0不存在该问题，因此torch选用了1.8.0
- 从profiling数据的op_statistic_0_1.csv看出影响性能的是Conv2D算子，ConfusionTransposeD，TransData算子，从op_summary_0_1.csv可以看出单个ConfusionTransposeD算子aicore耗时很多，以上为优化ConfusionTransposeD后的性能数据