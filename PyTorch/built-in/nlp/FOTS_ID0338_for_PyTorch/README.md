## 一、依赖
   
    NPU配套的run包安装   
    Python 3.7.5   
    PyTorch(NPU版本)  
    apex(NPU版本)  
    torch(NPU版本)


## 二、训练流程：

	1.安装环境
	2.开始训练  data_path为数据集icdar2015的路径 模型训练所需的resnet34-333f7ec4.pth文件也让需要放在该路径下
              bash ./test/train_full_1p.sh  --data_path=数据集路径   #精度训练



## 三、测试结果

    训练日志路径:

    /home/FOTS/test/output/device_id/