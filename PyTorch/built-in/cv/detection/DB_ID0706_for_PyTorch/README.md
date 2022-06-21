# DB模型使用说明

一、依赖
* NPU配套的run包安装(C20B030)
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* 安装geos，可按照环境选择以下方式：

  1. ubuntu系统：

     ```
     sudo apt-get install libgeos-dev
     ```

  2. euler系统：

     ```
     sudo yum install geos-devel
     ```

  3. 源码安装：

     ```
     wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
     bunzip2 geos-3.8.1.tar.bz2
     tar xvf geos-3.8.1.tar
     cd geos-3.8.1
     ./configure && make && make install
     ```
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision
    建议：Pillow版本是9.1.0  torchvision版本是0.6.0

- 安装python依赖包

  ```
  pip3.7 install -r requirements.txt
  ```

二、训练流程：

1、下载预训练模型MLT-Pretrain-Resnet50, [ Google Drive ]( https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG )，并将MLT-Pretrain-Resnet50 放在当前DB模型路径下 path-to-model-directory 文件夹中;

```
__ path-to-model-directory
  |__ MLT-Pretrain-ResNet50
```


2、 下载icdar2015数据集，放在文件夹datasets下;

```
__ datasets
  |__icdar2015
```


3、开始训练：
单卡训练流程：

```
	1.安装环境
	2.开始训练
              bash ./test/train_full_1p.sh  --data_path=./datasets            [ data_path为数据集路径，写到datasets，即data_path路径不包含icdar2015 ]
```

**注意**：如果发现打屏日志有报checkpoint not found的warning，请再次检查第二章节预训练模型MLT-Pretrain-Resnet50的配置，以免影响精度。

多卡训练流程：

```
	1.安装环境
                2.开始训练
              bash ./test/train_full_8p.sh  --data_path=./datasets             [ data_path为数据集路径，写到datasets，即data_path路径不包含icdar2015 ]
```



模型评估：

```
执行脚本 bash eval_precision.sh
```

