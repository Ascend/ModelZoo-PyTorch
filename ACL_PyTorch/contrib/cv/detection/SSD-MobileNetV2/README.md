$\color{red}{说明：删除线用于READ.md的说明，以下带有删除线的说明在README.md中需要删除}$  

# ReID-strong-baseline模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```


2.获取，修改与安装开源模型代码  
```
git clone https://github.com/qfgaohao/pytorch-ssd  
cd pytorch-ssd   
```

3.获取权重文件  

wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_object_detection/SSD-MobilenetV2/mb2-ssd-lite-mp-0_686.pth


4.数据集     
VOC2007 VOC2012

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| SSD-MobileNetV2 bs1  | 68.6% | 70.0% |  1024.394fps | 1253.516fps | 
| SSD-MobileNetV2 bs16 | 68.6% | 70.0% | 2109.705fps | 2124.212fps | 

 



