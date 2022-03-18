# SimCLR模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```
说明：PyTorch选用开源1.8.0版本



2.获取，修改与安装开源模型代码  
安装SimCLR
```shell
git clone https://github.com/sthalles/SimCLR
cd SimCLR
conda env create --name simclr --file env.yml
conda activate simclr
python run.py
```

3.获取权重文件  
  https://pan.baidu.com/s/18sZVnLoQpgIj_nuRpG-XnQ
  提取码：irpw 

4.数据集     
1. 获取CIFAR-10数据集
```
#Version：CIFAR-10 python version
```
2. 对压缩包进行解压到/root/datasets文件夹(执行命令：tar -zxvf cifar-10-python.tar.gz -C /root/datasets)，test_batch存放cifar10数据集的测试集图片，文件目录结构如下：
```
root
├── datasets
│   ├── cifar-10-batch-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
|   |   ├── data_batch_2
|   |   ├── data_batch_3
|   |   ├── data_batch_4
|   |   ├── data_batch_5
|   |   ├── test_batch
|   |   ├── readme.html
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
source env.sh
bash test/pth2om.sh  
bash test/eval_acc_perf.sh   
```
 **评测结果：**   
| 模型      | 在线推理精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| SimCLR bs1  | top1：65.625% | top1:65.069% |  2486.69fps | 4068.32fps | 
| SimCLR bs16 | top1：65.625% | top1:65.089% | 39876.3fps | 44084.8fps | 



