## 一、训练流程

单卡训练流程：

    1.安装环境
    2.修改参数device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
    3.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径  --epochs=训练周期数         # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径  --epochs=训练周期数  # 性能训练


多卡训练流程

    1.安装环境
    2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径  --epochs=训练周期数         # 精度训练
        bash ./test/train_performance_8p.sh  --data_path=数据集路径  --epochs=训练周期数  # 性能训练

## 二、Docker容器训练

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：


    ./docker_start.sh pytorch:b020 /train/peta /home/DeepMar

3.执行步骤一训练流程（环境安装除外）

### 三、测试结果

训练日志路径：网络脚本test下output文件夹内。例如：
      test/output/devie_id/train_${device_id}.log          # 训练脚本原生日志
      test/output/devie_id/DenseNet161_bs1024_8p_perf.log  # 8p性能训练结果日志
      test/output/devie_id/DenseNet161_bs1024_8p_acc.log   # 8p精度训练结果日志

训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。