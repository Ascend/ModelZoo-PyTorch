# IntraDA
  
- 参考实现：
```
url=https://github.com/feipan664/IntraDA.git
branch=master 
commit_id=070b0b702fe94a34288eba4ca990410b5aaadc4a
```

## IntraDA Detail

- 增加了混合精度训练
- 增加了多卡分布式训练
- 优化了loss在NPU上的计算效率

## Requirements

- CANN 5.0.2
- torch 1.5.0+ascend.post3.20210824
- apex 0.1+ascend.20210824
- 安装ADVENT
  ``` 
  cd IntraDA/ADVENT
  pip3 install -e .
  ```
- 下载[CityScapes数据集](https://www.cityscapes-dataset.com/downloads/)  
  在IntraDA/ADVENT目录下创建data文件夹，将数据集按照如下结构放入data目录：  
  ```
    |-- ADVENT
    |   |-- data
    |   |   `-- Cityscapes
    |   |       |-- gtFine
    |   |       `-- leftImg8bit
  ```
- 下载以下两个预训练模型：  
    ImageNet预训练模型：  
    https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/semantic_segmentation/IntraDA/DeepLab_resnet_pretrained_imagenet.pth  
    ADVENT warmup模型：  
    https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/semantic_segmentation/IntraDA/gta2cityscapes_advent.pth 
    在IntraDA/ADVENT目录下创建pretrained_models文件夹，将以上2个模型放入改文件夹，目录结构如下：
    ```
    |-- ADVENT
    |   |-- pretrained_models
    |   |   |-- DeepLab_resnet_pretrained_imagenet.pth
    |   |   `-- gta2cityscapes_advent.pth
    ```
- 生成训练用的伪标签及数据集分组文件： 
  ```
  cd IntraDA/entropy_rank/
  bash gen_color_mask_npu.sh
  ```


## Training

```bash
cd IntraDA/intrada

# 1p train perf 运行 500 step， 输出 performance_1p.log 文件
bash test/train_performance_1p.sh

# 8p train perf 运行 500 step， 输出 performance_8p.log 文件
bash test/train_performance_8p.sh

# 8p train full 完整训练并保存checkpoints，中间不会测试
bash test/train_full_8p.sh

# eval 测试8p训练保存的 checkpoints 得到精度信息
bash test/train_eval_8p.sh
```

## IntraDA training result

| mIoU     | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
|          | 2.7       | 1        | -        | O2       |
| 42.55    | 21        | 8        | -        | O2       |
