###  YOLOF模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 获取，修改与安装开源模型代码

```
git clone -b main https://github.com/megvii-model/YOLOF.git
cd YOLOF
git reset 6189487b80601dfeda89302c22abac060f977785 --hard

patch -p1 < ../YOLOF.patch
python3 setup.py develop
cd ..
```

3. 将权重文件YOLOF_CSP_D_53_DC5_9x.pth放到当前工作目录

4. 数据集

   获取COCO数据集，并重命名为coco，放到当前目录下的 YOLOF/datasets/ 文件夹内

5. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前工作目录

### 2. 离线推理

710上执行，执行时使npu-smi info查看设备状态，确保device空闲

```bash
bash test/pth2om.sh --batch_size=1
bash test/eval_acc_perf.sh  --batch_size=1
```

**评测结果：**

| 模型       | pth精度     | 710离线推理精度 | 性能基准 | 710性能  |
| ---------- | ----------- | --------------- | -------- | -------- |
| YOLOF bs1  | box AP:42.8 | box AP:42.8     | fps 14.7 | fps 51.1 |
| YOLOF bs16 | box AP:42.8 | box AP:42.8     | fps 18.3 | fps 60.9 |