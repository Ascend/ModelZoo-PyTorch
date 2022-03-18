## GAN模型PyTorch离线推理指导

### 1 环境准备

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

   ```python
   pip3.7 install -r requirements.txt
   ```

2. 数据集获取

   开源代码仓[点此进入](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py)没有提供模型测试相关的数据集和代码，这里采用自己设置的随机张量来测试模型的生成精度。  


3. 获取msame工具

   将编译好的msame工具放到当前目录  

### 2 离线推理

310上执行，执行时使用npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh 
bash test/eval_acc.sh
bash test/eval_bs1_perf.sh
bash test/eval_bs16_perf.sh
```



**评测结果：**

|  模型   | 官网pth精度 | 310离线推理精度 |  基准性能  |  310性能  |
| :-----: | :---------: | :-------------: | :--------: | :-------: |
| GAN bs1 | - |   -   | fps:12105.394 | fps: 9302.326|
| GAN bs16 |- | -     | fps:180332.895|fps: 136170.213|
