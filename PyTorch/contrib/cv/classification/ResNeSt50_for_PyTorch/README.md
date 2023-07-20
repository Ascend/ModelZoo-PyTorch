# ResNeSt50

该目录为ResNeSt50在ImageNet数据集上的训练与测试, 主要参考实现[zhanghang1989/ResNeSt](https://github.com/zhanghang1989/ResNeSt)

## ResNeSt50的相关细节

相较于GPU上的参考实现，该目录的ResNeSt50主要进行以下几点修改：
- 增加单卡训练部分，使模型能够在单卡上进行训练
- 对训练和Mixup模块针对NPU进行修改，Mixup模块具体见`module/utils.py`
- 添加混合精度相关代码，同时amp.initialize()中combine_grad设置为True
- SGD的实现使用apex.optimizers.NpuFusedSGD()

## 环境依赖

- 执行本样例前，请确保已安装有昇腾AI处理器的硬件环境，CANN包版本5.0.2
- 该目录下的实现是基于PyTorch框架，其中torch版本为1.5.0+ascend.post3，使用的混合精度apex版本为0.1+ascend
- `pip install -r requirements.txt`
  注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0
## 训练前准备

- 采用ImageNet(ILSVRC2012)数据集，数据集获取请参考：[PyTorch/ImageNet](https://github.com/pytorch/examples/tree/master/imagenet)
    - 解压数据集并使用`valprep.sh`脚本将验证集的图片移动到带有标签的子文件夹中： [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
    - 或运行`module/prepare_imagenet.py`文件  
    处理后得到的ImageNet数据集的目录结构如下所示(其中train与val的总和为146G，test可忽略)：
    ```bash
    # 注：类别的命名类似于'n01440764'格式
    ├── imagenet 
    │    ├──train
    │    │  ├──类别1──图片1、2、3、4            
    │    │  ├──类别2──图片1、2、3、4
    │    │  ├── ......
    │    │  └──类别n──图片1、2、3、4
    │    │                      
    │    ├──val  
    │    │  ├──类别1──图片1、2、3、4            
    │    │  ├──类别2──图片1、2、3、4
    │    │  ├── ......
    │    │  └──类别n──图片1、2、3、4
    ```
- 根据实际情况配置数据集地址和训练参数，配置文件格式请参考configs/Base-ResNeSt50.yaml
    - 以数据集地址的配置为例，需要修改配置文件中的DATA.ROOT
      ```
      DATA:
        ROOT: 'real_dataset_path'  # 根据情况修改为实际的数据集路径，推荐使用绝对路径
      ```

## 快速运行

模型的训练文件详见`train_npu.py`, 运行以下脚本能够进行单/多卡的训练和性能测试:

```bash
# 注: 以下类似于'real_config_path'的路径根据实际情况填写，请尽量使用绝对路径

# training 1p accuracy
bash ./test/train_full_1p.sh --config_path=/real_config_path/ResNeSt50_full_1p.yaml

# training 1p performance
bash ./test/train_performance_1p.sh --config_path=/real_config_path/ResNeSt50_performance_1p.yaml

# training 8p accuracy
bash ./test/train_full_8p.sh --config_path=/real_config_path/ResNeSt50_full_8p.yaml

# training 8p performance
bash ./test/train_performance_8p.sh --config_path=/real_config_path/ResNeSt50_performance_8p.yaml

# evaluating 8p
# 注：checkpoint_apex_final.pth作为参考，请以训练后产生的模型文件名为准
bash ./test/train_eval_8p.sh --config_path=/real_config_path/ResNeSt50_full_8p.yaml --checkpoint=/real_checkpoint_path/checkpoint_apex_final.pth

# training 1p finetune
# 注: finetune的脚本采用train_full_1p.sh，请在配置文件中修改num_classes等参数以实现迁移
bash ./test/train_full_1p.sh --config_path=/real_config_path/ResNeSt50_finetune_1p.yaml
```
训练以及验证的结果文件存储于
```bash
# 脚本中定义的输出文件地址
test/output/${devie_id}/train_${device_id}.log              # training detail log
test/output/${devie_id}/ResNeSt50_bs128_8p_perf.log         # 8p training performance result log
test/output/${devie_id}/ResNeSt50_bs128_8p_acc.log          # 8p training accuracy result log

# 脚本中的参数'--outdir'定义存储checkpoint和训练日志的地址，以train_full_8p.sh为例
output_8p_full/log_8p_full_O1_8.txt    # 8p训练的训练日志
output_8p_full/checkpint_apex_xxx.pth  # 8p训练的模型文件
```

## ResNeSt50的训练结果

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 388.38    | 1        | 1        | O1       |
| 78.808   | 3168.55   | 8        | 120      | O1       |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md