# SiamFC模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码  
```
代码地址：https://github.com/HonglinChu/SiamTrackers/tree/master/2-SiamFC/SiamFC-VID   
论文地址：https://arxiv.org/pdf/1606.09549.pdf
```
3.获取权重文件  

采用Ascend910上训练得到的权重文件[siamfc.pth](https://pan.baidu.com/s/1N3Igj4ZgntjRevsGA5xOTQ)，提取码：4i4l，放置于本代码仓./pth目录下

4.数据集     
[获取OTB2015数据集]([Visual Tracker Benchmark (hanyang.ac.kr)](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html))，并重命名为OTB，默认存放在/opt/npu目录下

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu  
```
> datasets_path参数根据数据集实际的存放位置而定，例如：OTB数据集存放位置为/opt/npu/OTB，则应设置参数--datasets_path=/opt/npu

 **评测结果：**   

|    模型    |               pth在线推理精度                |               310离线推理精度                |
| :--------: | :------------------------------------------: | :------------------------------------------: |
| siamfc_bs1 | success_score: 0.576  precision_score: 0.767 | success_score: 0.571  precision_score: 0.760 |

| 模型      | Benchmark性能 | 310性能    |
| :------: | :------:  | :------:  |
| exemplar_bs1 | 4240fps | 5677fps |
| search_bs1 | 738fps | 862fps |

> 由于该模型无法进行常规的离线测试，因而改为对测试集的每一帧进行测试，exemplar_bs1和search_bs1分别对应模型中的两个分支，它们所进行的操作不同。
>
> siamfc_bs1由exemplar_bs1和search_bs1这两部分组成，在评测精度时给出siamfc_bs1的精度，在评测性能时分别给出exemplar_bs1和search_bs1的性能。


