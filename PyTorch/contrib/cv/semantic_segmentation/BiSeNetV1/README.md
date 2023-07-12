# BiSeNetv1 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

旷视科技创新性地提出双向网络BiSeNet，它包含两个部分：Spatial Path (SP) 和 Context Path (CP)。顾名思义，这两个组件分别用来解决空间信息缺失和感受野缩小的问题。BiSeNet不仅实现了实时语义分割，还在特征融合模块和注意力优化模块的帮助之下，把语义分割的性能推进到一个新高度，从而为该技术的相关落地进一步铺平了道路。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation
  commit_id=b42c4877672f990733d5a704c907a27047229c61
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  |硬件|[1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)
  | NPU固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
  最新的Ascend-Pytorch版本为1.8.1。 选择MMSegmentation=0.10.0 和 mmcv=1.2.7，由于它们支持pytorch1.8.1。

+ 安装mmcv。

   首先，下载 [mmcv1.2.7](https://github.com/open-mmlab/mmcv/tree/v1.2.7) 到 `$YOURMMVCPATH`。 然后，复制 `mmcv_replace` 到 `$YOURMMVCPATH/mmcv`。

   检查numpy的版本为1.21.6。

```
# configure
cd $BiSeNetv1
source env_npu.sh

# copy
rm -rf $YOURMMVCPATH/mmcv  # $YOURMMVCPATH为用户指定的任意路径
mkdir mmcv
cp -r mmcv_replace/* $YOURMMVCPATH/mmcv/

# compile
cd $YOURMMVCPATH
export MMCV_WITH_OPS=1
export MAX_JOBS=8
python3 setup.py build_ext
python3 setup.py develop
pip list | grep mmcv

# check numpy version
pip show numpy
```

+ 返回$BiSenetv1文件夹。
```
cd $BiSenetv1
```


+ 权限配置。
```
chmod -R 777 ./
```

+ 删除 `mmcv_replace` 文件夹。
```
rm -rf mmcv_replace
```

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，选用的开源数据集为Cityscape，将数据集上传到服务器任意路径下并解压。

   Cityscape数据集目录结构参考如下所示。

   ```
   ├── Cityscape
         ├──leftImg8bit
              ├──train
                    │──图片1
                    │──图片2
                    │   ...       
              ├──val
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──gtFine  
              ├──train
                    │──图片1
                    │──图片2
                    │   ...       
              ├──val
                    │──图片1
                    │──图片2
                    │   ...              
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。





# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```
     
     启动单卡性能测试。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```
     启动8卡性能测试。

     ```
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  
     ```
   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --work-dir                //保存路径
   --load-from               //权重加载路径
   --resume-from             //恢复训练路径
   --no-validate             //取消训练中的验证      
   --seed                    //随机种子
   --device                  //设备类型，npu或gpu
   --amp                     //使用混合精度
   --loss-scale              //loss scale大小，输入-1为动态
   --opt-level               //混合精度类型
   --prof                    //使用prof评估模型训练表现
   --prof_test               //使用prof评估模型推理表现
   --warm_up_epochs          //热身
   多卡训练参数：
   --device_list             //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下Output文件夹中，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    |  mIoU |  FPS | Iterations | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-竞品 | -     |  1.5 | 750      |        - |
| 1p-NPU  | -     |  4.736 | 750      |       O2 |
| 8p-竞品 | 76.92 | 2.0 | 160000    |        - |
| 8p-NPU  | 77.14 | 3.096| 160000    |       O2 |



# 版本说明

## 变更

2022.11.1：首次发布。

## 已知问题

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md





