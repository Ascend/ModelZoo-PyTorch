## Training
一、依赖

    NPU配套的run包安装
    Python 3.7.5
    PyTorch(NPU版本)
    apex(NPU版本)
    torch(NPU版本)
    torchvision
    pillow
    注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision 建议Pillow版本是9.1.0 torchvision版本是0.6.0


二、训练流程
   data_path为imagenet数据集所在路径

单卡训练流程：

    1.安装环境  
    2.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径            # 精度训练


 
多卡训练流程

    1.安装环境
    2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径            # 精度训练



三、训练结果
    /home/Googlenet_ID0447_for_PyTorch/test/output/0/

