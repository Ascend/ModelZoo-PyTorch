# EDSR(Dynamic)模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
```shell
pip3.7 install -r requirements.txt
```

2.获取，修改与安装开源模型代码
```shell
mkdir workspace && cd workspace
git clone https://github.com/sanghyun-son/EDSR-PyTorch
cd EDSR-PyTorch
git reset 9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2 --hard
cd ../..
```

3.获取权重文件

获取[EDSR_x2预训练pth权重文件](https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt)

文件名：edsr_baseline_x2-1bc95232.pt

md5sum：e0a9e64cf1f9016d7013e0b01f613f68

4.数据集

获取B100数据集，解压到对应目录：

```shell
mkdir -p benchmark
mv B100 .benchmark
```

目录结构如下：

```shell
benchmark/
└── B100
    ├── HR
    └── LR_bicubic
```

## 2 离线推理

NPU上执行, 执行时使npu-smi info查看设备状态，确保device空闲：

${chip_name}可通过`npu-smi info`指令查看

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
bash test/pth2om.sh edsr_baseline_x2-1bc95232.pt ./Ascend${chip_name} # Ascend310P3
# 测试精度
bash test/eval_acc_perf.sh
# 测试性能
bash test/speed_test.sh
```

GPU机器上执行，执行时使用nvidia-smi查看设备状态，确保device空闲：

```
bash test/perf_g.sh
```

 **评测结果：**

精度：

| 模型         | Pth精度      | NPU离线推理精度 |
| :------:     | :------:     | :------:        |
| EDSR_Dynamic | PSNR: 32.352 | PSNR: 32.352    |
|--------------|--------------|-----------------|

性能：

| 模型         | Input shape   | NPU性能    | 基准性能    |
| :------:     | :------:      | :------:   | :------:    |
| EDSR_Dynamic | H:240,W:320   | 6.5420 fps | 17.5390 fps |
| EDSR_Dynamic | H:480,W:640   | 1.3742 fps | 4.5005 fps  |
| EDSR_Dynamic | H:720,W:1280  | 0.3874 fps | 1.5082 fps  |
| EDSR_Dynamic | H:1080,W:1920 | 0.1148 fps | 0.6677 fps  |
|--------------|---------------|------------|-------------|

