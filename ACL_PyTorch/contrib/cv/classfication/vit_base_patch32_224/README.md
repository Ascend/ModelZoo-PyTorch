# vit_base_patch32_224模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖
```
pip3.7 install -r requirements.txt
```

2.获取权重文件
从timm官方下载vit_base_patch32_224的权重文件

3.获取数据集

手动下载imagenet2012数据集并解压，并将 ILSVRC2012_validation_ground_truth.txt 和 meta.mat 与所有代码一起放置于同一目录下

4.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

将benchmark.x86_64或benchmark.aarch64放到当前目录

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --data_path=/root/datasets
```
**评测结果：**
| 模型      | pth精度  | 310精度  | 性能基准    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| vit_base bs1  | top1:80.724% | top1:80.714% |  339.818fps | 220.844fps | 
| vit_base bs16 | top1:80.724%  | top1:80.714% | 1172.531fps | 244.212fps | 

**备注：**
性能未达标