一、依赖
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* torch(NPU版本)
* torchvision

注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision
    建议：Pillow版本是9.1.0  torchvision版本是0.6.0



二、训练流程

单卡训练流程：

    1.安装环境
    2.修改参数device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
    3.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径  --epochs=训练周期数           # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径  --epochs=训练周期数    # 性能训练


​    
多卡训练流程

    1.安装环境
    2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径  --epochs=训练周期数           # 精度训练
        bash ./test/train_performance_8p.sh  --data_path=数据集路径  --epochs=训练周期数   # 性能训练


三、Docker容器训练

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：


    ./docker_start.sh pytorch:b020 /train/peta /home/DeepMar

3.执行步骤一训练流程（环境安装除外）

四、测试结果
训练日志路径：网络脚本test下output文件夹内。例如：
      test/output/devie_id/train_${device_id}.log          # 训练脚本原生日志
      test/output/devie_id/DenseNet169_bs1024_8p_perf.log  # 8p性能训练结果日志
      test/output/devie_id/DenseNet169_bs1024_8p_acc.log   # 8p精度训练结果日志

训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。

| DEVICE | FPS     | Npu_nums | Epochs | BatchSize | AMP  | ACC   |
| ------ | ------- | -------- | ------ | --------- | ---- | ----- |
| V100   | 370.932 | 1        | 90     | 128       | O2   | 75.06 |
| V100   | 2078.4  | 8        | 90     | 128*8     | O2   | 73.79 |
| NPU910 | 429.957 | 1        | 90     | 128       | O2   | 75.09 |
| NPU910 | 2220    | 8        | 90     | 128*8     | O2   | 73.90 |



