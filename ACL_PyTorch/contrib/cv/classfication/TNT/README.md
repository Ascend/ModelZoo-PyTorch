# TNT模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```


2.获取，修改与安装开源模型代码  
```
git clone https://github.com/huawei-noah/CV-Backbones.git   
cd CV-Backbones  
git checkout 7a0760f0b77c2e9ae585dcadfd34ff7575839ace
patch tnt_pytorch/tnt.py ../TNT.patch
cd .. 
cp CV-Backbones/tnt_pytorch/tnt.py .
```

3.获取权重文件  

tnt_s_81.5.pth.tar

4.数据集     
获取ImageNet 2012

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
| TNT bs1  | [rank1:81.5%](https://github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch) | rank1:81.5% |  89fps | 33fps | 
| TNT bs16 | [rank1:81.5%](https://github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch) | rank1:81.5% |  181fps| 83fps | 


