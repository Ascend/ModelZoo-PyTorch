# DynamicUNet_for_Pytorch

## 1. Reference
- repo: 
    -  dynamic_unet
        - https://github.com/fastai/fastai/blob/master/fastai/vision/models/unet.py
        - commit_id: 7ec403cd41079bc81d80d48de67f7ab2b8141929
    - awesome-semantic-segmentation-pytorch
        - https://github.com/Tramac/awesome-semantic-segmentation-pytorch
        - commit_id: 9d9e25da10e2299cf0c84b6e0be1c49085565d22 

## 2. Preparation
### 2.1 软件环境准备
    1. 安装 NPU 运行所需的driver，firmware，cann包，安装ascend—torch-1.8(当前模型仅在1.8上跑过), torch_npu, ascend-apex
    2. pip install -r requirements.txt

### 2.2 数据集准备
     - 下载[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
     - 将VOCdevkit放置当前目录或者软链到当前目录 

### 2.3 原始代码仓准备
- 当前仓clone了一份对应commit id的awesome-semantic-segmentation-pytorch代码仓到```./awesome-semantic-segmentation-pytorch```下，文件夹内部代码和github上开源代码完全一致没有修改，遵循开源代码仓相应协议和代码风格
- ```dynamic_unet.py```来自fastai，仅修改了输出样式以便兼容awesome-semantic-segmentation-pytorch的训练框架

## 3. 配置
- 使用了awesome-semantic-segmentation-pytorch仓readme中的示例配置，将模型更换为dynamicunet
  
## 4. 执行训练脚本
- 单卡性能训练执行 `bash test/train_performance_1p.sh`
- 8卡性能训练执行 `bash test/train_performance_8p.sh`
- 8卡精度训练执行 `bash test/train_full_8p.sh`

## 5. 结果

#### 单机

| mIoU      | FPS       | Npu_nums  | Epoch       |
| :------:  | :------:  | :------:  | :------:    |
| -         | 6.8       | 1         | -           |
| 0.535     | 48        | 8         | 50          |



