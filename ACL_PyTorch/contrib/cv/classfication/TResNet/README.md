# TResNet离线推理指导

## 1.环境准备
以路径${MODEL_ZOO_PATH}/contrib/ACL_PyTorch/Research/cv/classification/TResNet/作为当前目录

1.安装必备的依赖

```
pip3.7 install -r requirements.txt 
```

2.由于没有开源社区的权重，因此需要将训练得到的权重model_best.pth.tar放到当前目录

3.获取数据集imagenet，并且以${Path}/imagenet/val作为datasets_path，这将在下面用到

4.获取数据集imagenet的val_label.txt，并且以${Path}/val_label.txt作为val_label_path，这将在下面用到

5.获取benchmark工具

将benchmark.x86_64放到当前目录

6.（重要）请确保您的CANN环境为5.0.3.alpha003，以确保能获得最佳性能

7.（重要）由于有算子涉及到了TransposeD，因此请将以下shape添加至白名单[ 1,3, 224, 224],[ 1, 3, 56, 4, 56, 4],[ 1, 4, 4, 3, 56, 56],
    [ 16, 3, 224, 224],[ 16, 3, 56, 4, 56, 4], [ 16, 4, 4, 3, 56, 56]
8.请确保您能连接github以获取模型源码

## 2.离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/imagenet/val --val_label_path=/root/datasets/imagenet/val_label.txt
bash test/perf_g.sh
```

