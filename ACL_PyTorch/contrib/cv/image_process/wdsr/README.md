# WDSR模型PyTorch离线推理指导
## 1 环境准备
1. 安装必要的依赖
```
pip3.7 install -r requirements.txt
```

2. 获取开源模型代码
```
git clone https://github.com/ychfan/wdsr.git -b master 
cd wdsr
git reset --hard b78256293c435ef34e8eab3098484777c0ca0e10
cd ..
```

3. 获取权重文件
```
wget https://github.com/ychfan/wdsr/files/4176974/wdsr_x2.zip
unzip wdsr_x2.zip
rm wdsr_x2.zip
```

4. 获取数据集

[DIV2K数据集网址](https://data.vision.ee.ethz.ch/cvl/DIV2K/）

下载`Validation Data (HR images)`和`Validation Data Track 1 bicubic downscaling x2 (LR images)`两个压缩包，新建data/DIV2K文件夹，将两个压缩包解压至该文件夹中。

5. 获取[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

将benchmark.x86_64或benchmark.aarch64放到当前目录

## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
bash test/pth2om.sh 
bash test/eval_acc_perf.sh --datasets_path=./data/DIV2K
```
**评测结果：**

| 模型      | pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  |
| WDSR bs1  | psnr:34.76 | psnr:34.7537 | 6.8778fps | 5.936fps |
| WDSR bs8  | psnr:34.76 | psnr:34.7537 | 6.5681fps | 6.124fps |

**备注：**
 
- 在pth转换成onnx中，opset_version=9是因为,设置为11会DepthToSpace算子报错
- 在om的bs16版本在310设备上运行离线推理会out of memory。