# RotatE模型PyTorch离线推理指导

##  1  环境准备

- **1.1 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装**

```
pip3.7 install -r requirements.txt   
```

- **1.2 获取，修改与安装开源模型代码**

```
git clone https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding -b master 
cd KnowledgeGraphEmbedding
git reset --hard 2e440e0f9c687314d5ff67ead68ce985dc446e3a
cd ..
```
- **1.3 [获取权重文件](https://www.aliyundrive.com/drive/folder/616a7eb758db2df6ae8448e4b34fe570510ad216)**

- **1.4 开源模型代码里包含有数据集**

- **1.5 获取[msame工具](https://gitee.com/ascend/tools/tree/master/msame)和[benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)**

将msame和benchmark.x86_64（或benchmark.aarch64）放到当前目录

## 2 离线推理 

- **310上执行，执行时使npu-smi info查看设备状态，确保device空闲**

```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh
```

- **评测结果：**

| 模型        | pth精度                                                      | 310精度 | 性能基准       | 310性能         |
| ----------- | ------------------------------------------------------------ | ------- | -------------- | --------------- |
| RotatE-head  bs1<br>RotatE-tail    bs1| [**MRR:0.337**](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) | MRR:0.336   | 21.9065fps<br>21.9091fps | 99.3504fps<br>104.9432fps  |
| RotatE-head  bs16<br>RotatE-tail    bs16 | [**MRR:0.337**](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) | MRR:0.336   | 22.2017fps<br>22.1964fps | 119.9172fps<br>129.7252fps |
