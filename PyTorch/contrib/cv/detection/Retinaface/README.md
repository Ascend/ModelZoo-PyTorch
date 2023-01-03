# Retinaface for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

RetinaFace是一个强大的单阶段目标检测器，它利用联合监督和自我监督的多任务学习，可以在各种目标尺度上执行像素方面的目标定位。

- 参考实现：

  ```
  url=https://github.com/biubug6/Pytorch_Retinaface.git
  commit_id=b984b4b775b2c4dced95c1eadd195a5c7d32a60b
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
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
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1]( https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。


   - 下载[Wider Face](https://gitee.com/link?target=http%3A%2F%2Fshuoyang1213.me%2FWIDERFACE%2F)数据集，将数据集上传到服务器源码包根目录下的"data/"路径下并解压。

   - 从 [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) 下载注释文件`label.txt`，并放入对应的目录下。 
       
   - 确保 `wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat`,`wider_hard_val.mat` 文件在源码包根目录下的"widerface_evaluate/groud_truth"目录下。如果用户没有这些文件，请从https://github.com/biubug6/Pytorch_Retinaface.git 下载。如果用户已经下载了，请将`ground_truth`文件夹从"widerface_evaluate/groud_truth"复制到源码包根目录下的"widerface_evaluate/groud_truth"。
   
   - 在源码包根目录下的"widerface_evaluate/"路径下创建命名为`widerface_txt`的文件夹。



2. 数据预处理。

    用户将下载的label.txt文件放入对应路径下后，运行tools.py生成wider_val.txt文件。
    
    执行命令如下所示：
    ```
    python3.7 tools.py
    ```
   项目数据集目录结构参考如下所示：

   ```
     ./data/widerface/
       train/
         images/
         label.txt
       val/
         images/
         wider_val.txt
         label.txt
    ```   

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
     bash ./test/train_full_1p.sh --data_path=./data/widerface/train/label.txt    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data/widerface/train/label.txt   
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --val-data                          //val数据集路径
   --addr                              //主机地址
   --workers                           //加载数据进程数      
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.01
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   --gpu                               //使用的npu的id
   --rank                              //分布式训练的节点顺序
   --world-size                        //用于分布式训练的节点数
   多卡训练参数：
   --device_num                        //npu的数量
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Ap    |  FPS | Epochs | AMP_Type |Torch |
| ------- | :----- | :--- | ------ | -------: | ---- |
| 1p-NPU  | -     |  -   | 1      |        O2 |  1.5 |
| 1p-NPU  | -     | 1.185| 1      |       O2 |  1.8 |
| 8p-NPU  | Easy: 94.37 <br> Medium: 93.14 <br> Hard: 86.7 | 32.519 | 100    |        O2 |  1.5 |
| 8p-NPU  | Easy: 94.37 <br> Medium: 93.14 <br> Hard: 86.7 | 34.284 | 100    |        O2 |  1.8 |


# 版本说明

## 变更

2022.10.14：更新内容，重新发布。

2020.07.08：首次发布。

## 已知问题

无。










