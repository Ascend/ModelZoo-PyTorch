###  Pose Estimation UniFormer模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 获取，修改与安装开源模型代码

```
git clone -b main https://github.com/Sense-X/UniFormer.git
cd UniFormer
git reset e8024703bffb89cb7c7d09e0d774a0d2a9f96c25 --hard

patch -p1 < ../uniformer.patch
cd pose_estimation
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html
pip install -v -e .  # or python3 setup.py develop
cd ../..
```

3. 将权重文件[top_down_256x192_global_base.pth](https://drive.google.com/file/d/15tzJaRyEzyWp2mQhpjDbBzuGoyCaJJ-2/view?usp=sharing)放到当前工作目录

4. 数据集

   获取COCO数据集，并重命名为coco，放到/root/datasets目录

5. [获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)

   将msame文件放到当前工作目录

### 2. 离线推理

710上执行，执行时使npu-smi info查看设备状态，确保device空闲

```bash
bash test/pth2om.sh --batch_size=1
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/coco --batch_size=1
```

**评测结果：**

| 模型        | pth精度   | 710离线推理精度 | 性能基准  | 710性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| UniFormer bs1 | AP50=93.6 | AP50=93.5 | 88.914 fps | 162.601 fps |
| UniFormer bs16 | AP50=93.6 | AP50=93.5 | 116.939 fps | 277.441 fps |

**说明：**
使用COCO数据集的person_keypoints_val2017.json中的边界框标注，而非配置文件的COCO_val2017_detections_AP_H_56_person.json，因为后者中有非常多不含人的边界框，导致推理数据量过大且精度低