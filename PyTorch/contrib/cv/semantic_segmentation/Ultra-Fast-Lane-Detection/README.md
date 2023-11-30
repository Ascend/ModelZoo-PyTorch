# Ultra-Fast-Lane-Detection

这是Ultra-Fast-Lane-Detection模型在ResNet18模型上的训练部分，修改来自于[cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)



##  Requirements

- 安装提供NPU支持的PyTorch和混合精度Apex模块
- 安装必备的依赖包，命令：`pip3.7 install -r requirements.txt`
    注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0





## Training

训练模型需要使用所需的模型架构和 Tusimple 数据集的路径运行 train.py：

```
# 1p train perf
bash test/train_performance_1p.sh     --data_path=数据集路径

# 8p train perf
bash test/train_performance_8p.sh     --data_path=数据集路径

# 8p train full
bash test/train_full_8p.sh     --data_path=数据集路径

# 8p eval 
bash test/train_eval_8p.sh     --data_path=数据集路径

# finetuning
bash test/train_finetune_1p.sh     --data_path=数据集路径
```

注意：

- 所有训练脚本内部开头都包含train.py和tusimple.py。
- 具体路径需要使用者自己修改，同理测试脚本包含的test.py和tusimple.py也需要使用者根据具体情况修改路径
- train_performance_1p.log # 1p下测试性能的结果日志
- train_performance_8p.log # 8p下测试性能的结果日志
- train_full_8p.log # 8p 下完整训练的性能和精度的结果日志
- train_eval_8p.log # 8p 下验证精度的结果日志
- train_finetune_1p.log # 1p下fine-tuning的结果日志



### UFLD训练结果

| Metric | FPS  | Epochs | AMP_Type | Device |
| :----: | :--: | :----: | :------: | :----: |
|        | 153  |   1    |    O1    | 1p Npu |
| 94.98% | 1669 |  100   |    O1    | 8p Npu |
|        | 122  |   1    |    O1    | 1p Gpu |
| 95.46% | 832  |  100   |    O1    | 8p Gpu |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md