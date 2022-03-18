# GloRe模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt  
```  

2.获取，修改与安装开源模型代码

```

git clone https://github.com/facebookresearch/GloRe -b master 
cd GloRe
git reset --hard 9c6a7340ebb44a66a3bf1945094fc685fb7b730d
cd ..
```
3.[获取基于UCF101数据集训练出来的权重](https://ascend-pytorch-model-file.obs.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/GloRe/GloRe.pth)


4.[获取数据集UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)


5.[获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)

6.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

将benchmark.x86_64或benchmark.aarch64放到当前目录



## 2 离线推理
310上执行，执行时使npu-smi info查看设备状态，确保device空闲
``` 
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/UCF-101
```
  | 模型      | pth精度  | 310精度  | 基准性能    | 310性能    |
  | :------: | :------: | :------: | :------:  | :------:  | 
  | GloRe bs1  | top1:87.79% top5:98.02% | top1:87.77% top5:98.05% |  122.4380fps | 67.3636fps | 
 | GloRe bs16 | top1:87.79% top5:98.02% | top1:87.77% top5:98.05% |  148.0453fps | 71.7856fps | 
