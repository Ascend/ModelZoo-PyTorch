# RoBERTa模型PyTorch离线推理指导

环境版本：

*CANN:5.0.2*

*cuda:11.0*

*cudnn:8.2*

***TensoRT:8.0.0.3***

## 1.环境准备

1. 环境依赖+安装原始代码仓

   ```bash
   # 安装依赖
   pip3.7 install -r requirements.txt
   # 安装原仓代码
   git clone https://github.com/pytorch/fairseq.git
   cd fairseq
   git checkout c1624b27
   git apply ../roberta-infer.patch
   pip3.7 install --editable ./
   ```

2. 获取权重文件

   [RoBERTa模型pth权重文件](https://pan.baidu.com/s/1GZnnpz8fek2w7ARsZ0ujnA)，密码：x248

   解压后将checkpoint.pt文件放至`./checkpoints`目录下。

3. 获取数据集

   + 获取[SST-2官方数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)，解压至`./data`目录下，如`./data/SST-2`。

   + 对原始数据进行处理，生成可用数据文件，生成至`./data/SST-2-bin`：
   ```bash
   bash preprocess_GLUE_tasks.sh data/ SST-2
   ```

4. 获取benchmark工具

   将[benchmark.x86_64](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)，置于项目根目录

5. 获取msame工具

   根据[安装教程](https://gitee.com/ascend/tools/tree/master/msame)，安装msame，并将得到的可执行文件，置于项目根目录

## 2.执行推理

310上执行，执行时使用npu-smi info查看设备状态，确保device空闲

```bash
#设置环境
source env.sh

# pth转换为om
bash ./test/pth2om.sh
# 备注：生成的om文件在./outputs目录下

# 310使用msame测试精度性能
bash test/eval_acc_perf.sh msame
# 备注：使用msame对batch size 1和batch size 16进行测试，性能精度打印在屏幕上，结果文件在result文件夹下

# 310使用benchmark测试性能
bash test/eval_acc_perf.sh benchmark
# 备注：使用benchmark对batch size 1,4,8,16,32进行测试，性能结果perf文件保存在result文件夹下

# 基准环境性能测试
bash test/perf_benchmark.sh
# 备注：TensoRT版本为8.0.0.3，仅测试batch size 1
```

## 3.结果

精度性能

| 模型         | pth精度   | 310精度   | 基准性能  | 310性能  |
| ------------ | --------- | --------- | --------- | -------- |
| RoBERTa bs1  | acc:0.948 | acc:0.944 | 384.65fps | 12.01fps |
| RoBERTa bs16 | acc:0.948 | acc:0.943 |           | 98.49fps |