# FixRes模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```


2.获取，修改与安装开源模型代码  
```
git clone https://github.com/facebookresearch/FixRes.git -b main   
cd FixRes
git reset c9be6acc7a6b32f896e62c28a97c20c2348327d3 --hard
cd ..  
```

3.获取权重文件  

FixResNet50.pth

4.数据集     
获取ImageNet 2012

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲。测试时需下载imagenet_labels_fixres.json文件，并放在imagenet文件夹下。
```
# 生成om模型
bash test/pth2om.sh  

# om模型离线推理并测试
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| FixRes bs1  | [rank1:79.0%](https://github.com/facebookresearch/FixRes) | rank1:79.0% | 507fps | 785.208fps | 
| FixRes bs16 | [rank1:79.0%](https://github.com/facebookresearch/FixRes) | rank1:79.0% | 734fps | 788.566fps | 
