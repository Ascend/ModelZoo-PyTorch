# YoloV3 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

YOLOv3借鉴了YOLOv1和YOLOv2，在保持YOLO家族速度的优势的同时，提升了检测精度，尤其对于小物体的检测能力。YOLOv3算法使用一个单独神经网络作用在图像上，将图像划分多个区域并且预测边界框和每个区域的概率。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo
  branch=master
  commit_id=3e902c3afc62693a71d672edab9b22e35f7d4776
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
  
- 进入gitee.com/ascend/ModelZoo-PyTorch，下载zip文件，并将ModelZoo-PyTorch里面的YoloV3_ID1790_for_PyTorch压缩包传至服务器上并解压。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)或[1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* numpy 1.20.2
* PyTorch(NPU版本)
* apex(NPU版本)

## 准备数据集

   用户自行获取coco2017数据集，上传至服务器并解压，解压后目录如下所示：


   ```shell script
   ├── coco2017: #根目录
         ├──train2017 #训练集图片，约118287张
         ├──val2017 #验证集图片，约5000张
         │──annotations #标注目录             
   ```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. Build MMCV。
   
   ```
   cd ../
   git clone -b v1.2.7 --depth=1 https://github.com/open-mmlab/mmcv.git

   export MMCV_WITH_OPS=1
   export MAX_JOBS=8
   source ./test/env_npu.sh

   cd mmcv
   python3.7 setup.py build_ext
   python3.7 setup.py develop
   pip3.7 list | grep mmcv
   ```
   
   将mmcv_need目录下的文件替换到mmcv的安装目录下。
   安装完后执行以下命令：
   
   ```
   /bin/cp -f mmcv_need/_functions.py ../mmcv/mmcv/parallel/
   /bin/cp -f mmcv_need/builder.py ../mmcv/mmcv/runner/optimizer/
   /bin/cp -f mmcv_need/data_parallel.py ../mmcv/mmcv/parallel/
   /bin/cp -f mmcv_need/dist_utils.py ../mmcv/mmcv/runner/
   /bin/cp -f mmcv_need/distributed.py ../mmcv/mmcv/parallel/
   /bin/cp -f mmcv_need/optimizer.py ../mmcv/mmcv/runner/hooks/
   ```
   
   或者运行env_set.sh脚本，进行MMCV的安装

   ```
   bash env_set.sh
   ```
   
3. Build MMDET from source
   执行以下命令，安装mmdet
   ```
   cd YoloV3_for_PyTorch
   pip3.7 install -r requirements/build.txt
   pip3.7 install -v -e .
   pip3.7 list | grep mm
   ```
   

4. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     source ./test/env_npu.sh
     chmod +x ./tools/dist_train.sh
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --addr                              //主机地址
   --arch                              //使用模型，默认：densenet121
   --workers                           //加载数据进程数      
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.01
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
## hipcc检查问题
若在训练模型时，有报"which: no hipcc in (/usr/local/sbin:..." 的日志打印问题，
而hipcc是amd和nvidia平台需要的，npu并不需要。
建议在torch/utils/cpp_extension.py文件中修改代码，当检查hipcc时，抑制输出。
将 hipcc = subprocess.check_output(['which', 'hipcc']).decode().rstrip('\r\n')修改为
hipcc = subprocess.check_output(['which', 'hipcc'], stderr=subporcess.DEVNULL).decode().rstrip('\r\n')

## invalid pointer问题
在Ubuntu、x86服务器上训练模型，有时会报invalid pointer的错误。
解决方法：去掉scikit-image这个依赖，pip3 uninstall scikit-image

## 单卡训练时，如何指定使用第几张卡进行训练
1. 修改 tools/train.py脚本
 将133行，cfg.npu_ids = range(world_size) 注释掉
 同时在meta['exp_name'] = osp.basename(args.config)后添加如下一行
 torch.npu.set_device(args.npu_ids[0])
2. 修改train_1p.sh
在PORT=29500 ./tools/dist_train.sh configs/yolo/yolov3_d53_320_273e_coco.py 1 --cfg-options optimizer.lr=0.001 --seed 0 --local_rank 0 后增加一个配置参数
--npu_ids k （k即为指定的第几张卡）

## 报No module named 'mmcv._ext'问题
在宿主机上训练模型，有时会报No module named 'mmcv._ext'问题，或者别的带有mmcv的报错。
解决方法：这一般是因为宿主机上安装了多个版本的mmcv，而训练脚本调用到了不匹配yolov3模型使用的mmcv，因此报mmcv的错误。
为了解决这个问题，建议在启动训练脚本前，先导入已经安装的符合yolov3模型需要的mmcv路径的环境变量。
export PYTHONPATH=mmcv的路径:$PYTHONPATH
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-竞品 | -     |  8 | 273      |        - |
| 1p-NPU  | -     |  30 | 273      |       O2 |
| 8p-竞品 | 27 | 41 | 273    |        - |
| 8p-NPU  | 26.5 | 243 | 273    |       O2 |



# 版本说明

## 变更

2022.9.28：更新内容，重新发布。

2022.3.18：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。











