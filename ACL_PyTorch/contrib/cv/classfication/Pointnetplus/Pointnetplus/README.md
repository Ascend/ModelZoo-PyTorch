# PointNet++ Onnx模型端到端推理指导

## 1. 环境准备
1.1.安装必要的依赖

```shell
pip3.7 install -r requirements.txt
```

1.2.获取，修改与安装开源模型代码

```shell
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch models
cd models
git checkout e365b9f7b9c3d7d6444278d92e298e3f078794e1
patch -p1 < ../models.patch
cd ..
```

1.3. 获取权重文件

pth采用开源仓自带的权重，权重位置：
```shell
./models/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth
```

1.4. 数据集

[测试集](https://shapenet.cs.stanford.edu)

1.5. 获取离线推理工具

[benchmark](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)
[msame](https://gitee.com/ascend/tools/tree/master/msame)

## 2. 离线推理

2.1 模型转换&&精度测试
在310上执行，执行时采用npu-smi info查看设备状态，确保device空闲

```shell
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets
```

2.2 性能测试

gpu测速，在对应gpu设备上执行，执行时采用nvidia-smi查看设备状态，确保device空闲

```shell
bash test/perf_g.sh
```
npu测速，在对应npu设备上执行，执行时采用npu-smi info查看设备状态，确保device空闲

```shell
./benchmark.x86_64 -round=100 -om_path=Pointnetplus_part1_bs1.om -device_id=0 -batch_size=1
./benchmark.x86_64 -round=100 -om_path=Pointnetplus_part2_bs1.om -device_id=0 -batch_size=1
./benchmark.x86_64 -round=100 -om_path=Pointnetplus_part1_bs16.om -device_id=0 -batch_size=16
./benchmark.x86_64 -round=100 -om_path=Pointnetplus_part2_bs16.om -device_id=0 -batch_size=16
```

**评测结果:**

| 模型             | batch_size | 官网pth精度                                 | 310离线推理精度                         | 基准性能 | 310性能 |
|------------------|------------|---------------------------------------------|-----------------------------------------|----------|---------|
| PointNet++_part1 | 1          | -                                           | -                                       | 2997fps  | 5308fps |
| PointNet++_part2 | 1          | Instance Acc: 0.928964, Class Acc: 0.890532 | Instance Acc: 0.9263, Class Acc: 0.8877 | 2571fps  | 4105fps |
| PointNet++_part1 | 16         | -                                           | -                                       | 3468fps  | 5968fps |
| PointNet++_part2 | 16         | -                                           | Instance Acc: 0.9245, Class Acc: 0.8854 | 3670fps  | 3730fps |
