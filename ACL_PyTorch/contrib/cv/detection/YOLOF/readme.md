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

5. [获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)

   将msame文件放到当前工作目录

### 2. 离线推理

310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```bash
bash test/pth2om.sh --batch_size=1 --soc_version=Ascend${chip_name}
bash test/eval_acc_perf.sh  --batch_size=1
```

**评测结果：**

| 模型       | pth精度     | 310P离线推理精度 | 310P性能    |
| ---------- | ----------- | --------------- | ---------- |
| YOLOF bs1  | box AP:50.9 | box AP:51.0     | fps 27.697 |
| YOLOF bs16 | box AP:50.9 | box AP:51.0     | fps 38.069 |