# DQN模型Pytorch离线推理指导

### 1 环境准备
**1.1 安装必要的依赖**
- pip install -r requirements.txt
- conda install mpi4py
- git clone https://github.com/openai/baselines.git  
  cd baselines  
  pip install -e '.[all]'  
  (要求tensorflow必须在1.14及以上)

**1.2 获取权重文件**   
本代码仓已提供：dqn.pth

**1.3 获取数据集**  
该模型没有原始输入的数据集，故而将在线推理的state输出保存作为数据集。将在线推理生成的state输出保存为pt文件，并将pt文件转成bin。
- `bash test/get_dataset.sh`

### 2 离线推理
- **310上执行，执行时使npu-smi info查看设备状态，确保device空闲**

执行如下脚本生成om模型 
```
bash test/pth2onnx.sh 

bash test/pth2om.sh  
```

执行如下脚本进行离线推理的精度测试和性能测试  
1-2行是使用msame工具离线推理  
5-9行是使用benchmark工具测试性能  
12-13行将离线推理结果和在线推理比较  
15-22输出Ascend310推理的性能结果
```
bash test/eval_acc_perf.sh
```

- **评测结果：**

| 模型        | 310精度/pth精度 | 性能基准（t4）   |  310性能         |
| -----------|---------------| -------------- | --------------- |
| DQN bs1    |    1.00     |  16878.98   |        13092.04  |