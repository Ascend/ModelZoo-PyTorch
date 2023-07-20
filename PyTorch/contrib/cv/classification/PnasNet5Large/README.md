# PnasNet5Large

该目录为在PnasNet5Large在ImageNet数据集上训练的脚本实现，主要参考 [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch#pnasnet)。

## PnasNet5Large实现细节

相比于GPU，NPU实现做了以下修改：
* amp initialize添加combine_grad=True
* 修改了混合精度计算写法，见crossentropy.py 
* 训练使用的learning rate由0.32改为0.4
* 取消zero pad操作，见pnasnet.py的line 93,96,141,144,195

## 依赖

- 安装 PyTorch ([pytorch.org](http://pytorch.org))
  * `pip3 install -r requirements.txt`
  注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision,建议Pillow版本是9.1.0 torchvision版本是0.6.0
- 下载ImageNet数据集， 下载地址： http://www.image-net.org/
  * 然后使用该脚本将验证集的图片移动到带有标签的子文件夹中： [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## 训练

通过运行 `imagenet_fast.py` 来使用ImageNet数据集训练模型，使用以下命令运行训练脚本：

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# train 1p finetune
# 注意：需要根据实际情况将训练脚本train_finetune_1p.sh中的resume变量修改为checkpoint文件的具体路径
#      并且不能放到 ./test/output/0/ 这个目录下
bash ./test/train_finetune_1p.sh --data_path=real_data_path

# evaluate
# 注意：需要根据实际情况将训练脚本train_eval_8p.sh中的resume变量修改为checkpoint文件的具体路径
#      并且不能放到 ./test/output/0/ 这个目录下
bash ./test/train_eval_8p.sh --data_path=real_data_path
```

日志存储路径:
```
    test/output/device_id/nohup.out                          # console log
    test/output/devie_id/train_${device_id}.log              # training detail log
    test/output/devie_id/WideReesnet50_2_bs8192_8p_perf.log  # 8p training performance result log
    test/output/devie_id/WideReesnet50_2_bs8192_8p_acc.log   # 8p training accuracy result log
```


## PnasNet5Large1.5 训练结果

| Acc@1    | FPS       | Npu_nums | Epochs     | AMP_Type |
| :------: | :------:  | :------: | :------:   | :------: |
| -        | 45       | 1        | 1(1000step) | O2       |
| 78.842   | 353      | 8        | 90          | O2       |

## PnasNet5Large1.8 训练结果

| Acc@1    | FPS       | Npu_nums | Epochs     | AMP_Type |
| :------: | :------:  | :------: | :------:   | :------: |
| -        | 69       | 1        | 1(1000step) | O2       |
| 79.625   | 353      | 8        | 90          | O2       |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md