# RCF模型PyTorch离线推理指导

## 1 环境准备 

### 1.1 安装必要的依赖
测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip install -r requirements.txt  
```

### 1.2 获取，修改与安装开源模型代码  

```
git clone https://github.com/meteorshowers/RCF-pytorch
cd RCF-pytorch
git reset --hard 6e039117c0b36128febcbe2609b27cc89740a3a8
cp ../RCF.diff ./
git apply --check RCF.diff
git apply RCF.diff
cd ..  
```

### 1.3 获取权重文件  

下载相应的模型文件，下载链接请参考相应的指导书，并放入前面已下载好的pytorch参考实现代码路径RCF-pytorch目录下。

### 1.4 获取数据集
在主目录新建data文件夹
本模型使用BSDS500数据，从官网获取BSDS500，获取链接请参照指导书，解压为BSR文件夹，并放入主目录中的data文件夹下 

### 1.5 获取评测方法代码 
```bash
git clone https://github.com/Walstruzz/edge_eval_python.git
cd edge_eval_python
git reset --hard 3e2a532ab939f71794d4cc3eb74cbf0797982b4c
cd ..
```

### 1.6 获取benchmark
[benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/) ，将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 
由于下载的RCF权重文件只能在gpu上进行转换模型，所以需要先在gpu上转完模型，然后再将onnx传到310上进行转换成om模型
### 2.1 T4精度及性能
T4上执行，执行时使用nvidia-smi查看设备状态，确保device空闲
- 转成onnx
```bash
bash test/pth2onnx.sh
```

- 输出T4精度
```
bash test/eval_acc_gpu.sh datasets_path data/BSR/BSDS500/data/images/test
```
- 输出T4性能
```
bash test/perf_gpu.sh
```

### 2.2 310精度及性能
310上执行，执行时使用npu-smi info查看设备状态，确保device空闲，输出310相应的精度和性能

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash test/onnx2om.sh 
bash test/eval_acc_perf.sh datasets_path data/BSR/BSDS500/data/images/test batch_size 1 device_id 0
```


 **评测结果：**   
| 模型      | pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  |
| RCF bs1  | ODS：79.8% OIS：81.7% | ODS：79.8% OIS：81.7% |  196fps | 89fps |
| RCF bs16  | ODS：79.8% OIS：81.7% | ODS：79.8% OIS：81.7% |  193fps | 86fps |