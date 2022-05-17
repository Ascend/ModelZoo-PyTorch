###  YOLOX模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip3.7 install -r requirements.txt
```

2. 获取，修改与安装开源模型代码

```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
git reset c9d128384cf0758723804c23ab7e042dbf3c967f --hard
```

3. 在https://github.com/Megvii-BaseDetection/YOLOX 界面下载YOLOX-s对应的weights， 名称为yolox_s.pth。

4. 数据集

   获取COCO数据集，并重命名为COCO，放到/root/datasets目录

5. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前工作目录

### 2. 离线推理

710上执行，执行时使npu-smi info查看设备状态，确保device空闲

```bash
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```

**评测结果：**

| 模型        | pth精度   | 710离线推理精度 | 性能基准  | 710性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| yolox-s | map:40.5% | map:40.1%    |          | 950fps  |



