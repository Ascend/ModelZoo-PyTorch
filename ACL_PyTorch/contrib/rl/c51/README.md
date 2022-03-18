# C51模型Pytorch离线推理指导

### 1 环境准备
**1.1 安装必要的依赖**
- pip install -r requirements.txt
- conda install mpi4py
- git clone https://github.com/openai/baselines.git  
  cd baselines  
  pip install -e '.[all]'  
  (要求tensorflow必须在1.14及以上)

**1.2 获取，修改开源模型代码**  
在已经下载推理代码的前提下，进入模型代码仓目录
- git clone https://github.com/ShangtongZhang/DeepRL
- cd DeepRL
- git apply ../c51-infer-update.patch

**1.3 获取权重文件**   
本代码仓已提供：c51.model、c51.stats

**1.4 获取数据集**  
该模型没有原始输入的数据集，故而将在线推理的输入输出保存作为数据集和标签。将在线推理生成的输入输出保存为pt文件，并将输入pt文件转成bin。
- `bash test/get_dataset.sh`

### 2 离线推理
- **310上执行，执行时使npu-smi info查看设备状态，确保device空闲**

执行如下脚本生成om模型  
1-12行是pth2onnx  
14-25行是onnx2om
```
bash test/pth2om.sh  
```

执行如下脚本进行离线推理的精度测试和性能测试  
1-2行是使用msame工具离线推理  
5-15行是使用benchmark工具测试性能  
18-19行将离线推理结果和在线推理比较  
21-30输出Ascend310推理的性能结果
```
bash test/eval_acc_perf.sh
```

- **评测结果：**

| 模型        | 310精度/pth精度 | 性能基准（t4）   |  310性能         |
| -----------|---------------| -------------- | --------------- |
| C51 bs1    |    0.996      |  16141.45    |        13117.28  |



