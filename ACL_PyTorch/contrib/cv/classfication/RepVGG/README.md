# RepVGG模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码  
```
git clone https://github.com/DingXiaoH/RepVGG   
cd RepVGG  
git reset 9f272318abfc47a2b702cd0e916fca8d25d683e7 --hard
cd ..  
``` 

3.获取权重文件  
[RepVGG-A0-train.pth](https://pan.baidu.com/s/1nCsZlMynnJwbUBKn0ch7dQ), and the access code is "rvgg"

4.数据集     
[获取imagenet]该模型使用ImageNet官网的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt。

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| RepVGG bs1  | [acc1:72.418](https://github.com/DingXiaoH/RepVGG) | acc1:72.14 |  1447fps | 2199fps | 
| RepVGG bs16 | [acc1:72.418](https://github.com/DingXiaoH/RepVGG) | acc1:72.14 | 4164fps | 4284fps | 
