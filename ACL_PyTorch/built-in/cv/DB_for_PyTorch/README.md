# DB模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码  
```
git clone https://github.com/MhLiao/DB   
cd DB  
git reset 4ac194d0357fd102ac871e37986cb8027ecf094e --hard
patch -p1 < ../db.diff  
cd ..  
```

3.获取权重文件  
```
wget https://github.com/MhLiao/DB/ic15_resnet50 -O DB/ic15_resnet50
```

4.数据集    
本模型数据集需要放在DB/  
datasets/icdar2015/  
├── test_gts  
├── test_images  
├── test_list.txt  
├── train_gts  
└── train_list.txt  

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到DB/  

6.因包含dcn自定义算子，去除对onnx的检查  
将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py的_check_onnx_proto(proto)改为pass  

## 2 离线推理 

npu上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
bash test/pth2om.sh  Your_Soc_Version
bash test/eval_acc_perf.sh --datasets_path=`pwd`/DB/datasets  
```
Your_Soc_version是你的npu型号，目前可选值为Ascend310和Ascend310P

 **评测结果：**   
| 模型      | 官网pth精度  | 310离线推理精度  | gpu性能    | 310性能    | 310P |
| :------: | :------: | :------: | :------:  | -------- | -------- |
| DB bs1  | precision : 91.3 recall : 80.3 fmeasure : 85.4 | 0.886823 0.803563 0.843142 |  8.53fps | 10.9282fps | 15.21fps |
| DB bs16 | precision : 91.3 recall : 80.3 fmeasure : 85.4 | 0.886823 0.803563 0.843142 | 7.240fps | 11.22672fps | 22.79fps |
| DB bs32 | - | 0.886823 0.803563 0.843142 | - | - | 22.46fps |
| DB bs64 | - | 0.886823 0.803563 0.843142 | - | - | 23.14fps |
备注：精度问题参见[issue](https://github.com/MhLiao/DB/issues/250)，修改代码后在线推理重新测评pth精度，可以看出om与pth精度一致  


