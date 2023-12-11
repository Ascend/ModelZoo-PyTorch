# Detr模型-推理指导

- [概述](##概述)

- [推理环境准备](##推理环境准备)

- [快速上手](##快速上手)
  
  - [获取源码](##获取源码)
  - [准备数据集](##准备数据集)
  - [模型推理](##模型推理)

- [模型推理性能](##模型推理性能)

- [配套环境](##配套环境)

## 概述

DETR是将目标检测视为一个集合预测问题（集合其实和anchors的作用类似）。由于Transformer本质上是一个序列转换的作用，因此，可以将DETR视为一个从图像序列到一个集合序列的转换过程。

- 参考论文：Carion, Nicolas, et al. "End-to-end object detection with transformers." European Conference on Computer Vision. Springer, Cham, 2020.

- 参考实现：

  ```
  url=https://github.com/facebookresearch/detr
  branch=master
  commit_id=b9048ebe86561594f1472139ec42327f00aba699
  model_name=DETR
  ```

  适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch
  tag=v.0.4.0
  code_path=ACL_PyTorch/contrib/cv/detection
  ```

  通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据

- 输入数据

  | 输入数据  | 数据类型 |       大小        | 数据排布格式  |
  | -------- | -------- | ---------------- | ------------ |
  | input    | RGB_FP32 |      多尺度       |      ND      |


- 输出数据

  |  输出数据    |      大小     | 数据类型  | 数据排布格式  |
  |  --------   |  -----------  | -------- | ------------ |
  | pred_boxes  |  1 x 100 x 92 | FLOAT32  |    NCHW      |
  | Pred_logits |  1 x 100 x 4  | FLOAT32  |    NCHW      |


## 推理环境准备[所有版本]

## 获取源码

1. 获取源码。

   ```
   上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。 
   ├── detr.patch               #修改detr补丁
   ├── detr_excute_omval.py     #torch_aie模型编译脚本
   ├── cal_acc.py               #精度脚本
   ├── LICENSE
   ├── README.md                #readme文档
   ├── requirements.txt         #所需环境依赖
   └── transformer.py           #图片前处理
   ```

## 获取开源代码仓。

   ```
   在已下载的源码包根目录下，执行如下命令。
   git clone https://github.com/facebookresearch/detr.git
   cd detr
   git checkout b9048ebe86561594f1472139ec42327f00aba699
   patch -p1 < ../detr.patch
   cd ..
   ```

## 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | -                                                       |
| Python                | 3.9.0           |                                                           
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |-
| Ascend-cann-torch-aie | -               
| Ascend-cann-aie       | -               
| 芯片类型                  | Ascend310P3     | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
## 安装CANN包

 ```
 chmod +x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run 
./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run --install
 ```
下载Ascend-cann-torch-aie和Ascend-cann-aie得到run包和压缩包
## 安装Ascend-cann-aie
 ```
  chmod +x Ascend-cann-aie_6.3.T200_linux-aarch64.run
  ./Ascend-cann-aie_6.3.T200_linux-aarch64.run --install
  cd Ascend-cann-aie
  source set_env.sh
  ```
## 安装Ascend-cann-torch-aie
 ```
 tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
 pip3 install torch-aie-6.3.T200-linux_aarch64.whl
 ```

## 安装其他依赖
```
pip3 install pytorch==2.0.1
pip3 install torchVision==0.15.2
```


## 准备数据集

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   
   本模型支持coco val 5000张图片的验证集。请用户根据代码仓readme获取数据集，上传数据集到代码仓目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到coco val2017.zip验证集及“instances_val2017.json”数据标签。
   
   [数据集链接](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
   
   数据目录结构请参考：
   ```
   coco_data    
   ├── annotations    
   └── val2017
   ```

2. torch_aie模型编译。
   

   1. 获取权重文件。 
    从源码包中获取权重文件：[detr.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/22.1.30/ATC%20Detr%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip)，将权重文件放入model文件夹。
   ```
    mkdir model
    cp ./detr.pth ./model
   ```
   数据预处理将原始数据集转换为模型输入的数据。
   执行“compile_models.py”脚本，完成9中尺度的torch_aie模型编译。

   ```
   mkdir compiled_models
   python3 compile_models.py --compiled_output_dir compiled_models
   ```

## 模型推理
   ```
   python3 cal_acc.py --coco_path=coco_data  --batch_size=1 --torch_aie_model_dir=./compiled_models
   ```
# 模型推理性能&精度
## 性能

|   Model   |   Batch Size    | 310P (FPS/Card) |
| --------- | --------------- |-----------------|
|   Detr    |        1        | 45.36           |
|   Detr    |        4        | 43.14           |

## 精度

- Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
- Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
- Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
- Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
- Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
- Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.613
- Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
- Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.529
- Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.570
- Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.305
- Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.626
- Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.805

|   Model   |   数据集    | 精度(MAP) |
| --------- | --------------- |---------|
|   Detr    |        coco        | 41.6    |