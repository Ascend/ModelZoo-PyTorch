# ViT-NPU 

# 预训练模型
将预训练模型ViT-B_16.npz放到当前路径下

## 数据集
将cifar-10-batches-py目录拷贝到当前路径的data目录下

## 启动训练
执行脚本：

```shell
bash run_vitbase_1p.sh
```

若需要修改执行训练的train_batch_size、num_steps，请自行修改run_vitbase_1p.sh中的参数

## 查看训练日志
tail -f查看当前目录下存放的训练日志vitbase.log



