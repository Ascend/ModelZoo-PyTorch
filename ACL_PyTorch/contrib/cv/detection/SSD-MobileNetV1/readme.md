# SSD MobileNetV1模型PyTorch离线推理


## 一. 环境准备
### 1.通过requirements.txt 安装必要依赖
```bash
pip3 install -r requirements.txt
```

### 2. 获取开源模型代码及开源权重
开源仓库：
```bash
cd SSD-MobileNetV1
git clone https://github.com/qfgaohao/pytorch-ssd.git -b master
cd pytorch-ssd
git reset f61ab424d09bf3d4bb3925693579ac0a92541b0d --hard
cd ..
```
预训练权重：
```bash
wget https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
```
下载后重命名为mobilenet-v1-ssd.pth

### 3. 获取测试数据集
这里使用VOC2007的测试集作为测试数据集
URL：https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
解压后获得VOCdevkit文件，并以VOCdevkit其作为数据集路径.

数据标签：
```bash
wget https://storage.googleapis.com/models-hao/voc-model-labels.txt
```

### 4. 获取benchmark工具
URL：https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/
将benchmark.x86_64放在当前目录下

## 二. 离线推理
310上执行，执行时使npu-smi info查看设备状态，确保device空闲
获取om文件：
```bash
bash test/pth2om.sh
```

进行推理，并评测准确率：
```bash
bash test/eval_acc_perf.sh --datasets_path=~/datasets/VOCdevkit 
```


### 评测结果：

模型         |     官网pth精度     |    310离线推理精度   |   性能基准   |   310性能
---- | ----- | ------ | ---- | ----- |
SSDMobileNet bs1   |     67.5%          |       69.3%          |   1262fps   |    1213fps    |
 SSDMobileNet bs16  |     67.5%          |       69.3%          |   2925fps   |    2330fps   |


