### PyTorch模型RFCN使用说明

note
- please download lib/pycocotools from origin repo if necessary:
- https://github.com/RebornL/RFCN-pytorch.1.0/tree/master/lib/pycocotools


## 环境准备
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* (可选)参考《Pytorch 网络模型移植&训练指南》6.4.2章节，配置cpu为性能模式，以达到模型最佳性能；不开启不影响功能。
* 安装依赖包
```
$ apt-get install -y numactl
$ pip install -r requirements.txt
```

## 数据集及预训练权重准备
* 新建data目录
```
mkdir data
cd data
```

* 下载PASCAL_VOC2007数据集
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

* 解压数据集
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```


* 新建预训练权重放置目录
```
mkdir pretrained_model
cd pretrained_model
```

* 将预训练权重resnet101_rcnn.pth放入当前目录

## 开启训练
* 设置device日志等级为error，确保性能最佳
```
/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 1 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 2 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 3 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 5 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 6 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 7 -g error
```


* 单p训练

	1.安装环境
	2.开始训练
              bash ./test/train_full_1p.sh  --data_path=数据集路径 

* 8p训练
  
	1.安装环境
	2.开始训练
              bash ./test/train_full_8p.sh  --data_path=数据集路径 
