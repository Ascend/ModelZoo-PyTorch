# CTPN模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码  
```
git clone https://github.com/CrazySummerday/ctpn.pytorch -b master   
cd ctpn.pytorch  
git reset 99f6baf2780e550d7b4656ac7a7b90af9ade468f --hard
cd ..  
```

3.获取权重文件  

在git clone的代码仓ctpn.pytorch中的weight文件夹中自带相应的ctpn.pth权重文件

4.获取数据集及相应评测方法代码 
在本目录新建data文件夹
获取ICDAR2013，获取链接见指导书，解压为Challenge2_Test_Task12_Images文件夹，并放入本目录data文件夹下
获取评测方法代码，获取链接见指导书，并解压为script文件夹放入本目录下
以上数据集以及代码的获取如果无法直接下载则需要注册

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

### 2.1 310精度及性能
310上执行，执行时使用npu-smi info查看设备状态，确保device空闲，输出310相应的精度和性能
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=./data/Challenge2_Test_Task12_Images  
```
### 2.2 T4精度及性能
T4上执行，执行时使用nvidia-smi查看设备状态，确保device空闲
- 输出T4精度
```
bash test/eval_acc_gpu.sh
```
- 输出T4性能
```
bash test/perf_gpu.sh
```

 **评测结果：**   
| 模型      | pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  |
| CTPN bs1  | precision:87.41% recall:75.60% hmean:81.08% | precision:86.84% recall:75.05% hmean:80.52% |  70.12fps | 91.26fps |