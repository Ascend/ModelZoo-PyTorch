
# 3D_ResNet_ID0421_for_PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

3D_ResNet是一个经典的动作识别网络。结构包括卷积层池化层等，训练好的模型输入图片后可以识别出人的动作，准确率高。

- 参考实现：
  ```
  url=https://github.com/kenshohara/3D-ResNets-PyTorch.git
  commit_id=540a0ea1abaee379fa3651d4d5afbd2d667a1
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取Hmdb51数据集。

* 下载视频和训练/测试拆分 [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)。

   

2. 数据预处理（按需处理所需要的数据集）。
* 将avi转化为jpg ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path hmdb51
```

* 使用类似于 ActivityNet 的 json 格式生成注释文件 ```util_scripts/hmdb51_json.py```
  

```bash
python -m util_scripts.hmdb51_json annotation_dir_path jpg_video_dir_path dst_json_path
```
 > **说明:**
 >```annotation_dir_path``` 包括 brush_hair_test_split1.txt, ...
* 预处理后ResNet3D数据集目录结构参考如下所示。
   ```
   ├── data
        ├──hmdb51_jpg
              ├──brush_hair
                    │──April_09_brush_hair_u_nm_np1_ba_goo_0
                        |——image_00001.jpg
                        |——image_00002.jpg
                        |   ... 
                    │──April_09_brush_hair_u_nm_np1_ba_goo_1
                        |——image_00001.jpg
                        |——image_00002.jpg
                        |   ... 
                    │   ...       
              ├──cartwheel
                    │──Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_fr_bad_3
                        |——image_00001.jpg
                        |——image_00002.jpg
                        |   ...  
                    │──Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_fr_bad_4
                        |——image_00001.jpg
                        |——image_00002.jpg
                        |   ... 
                    │   ...   
              ├──...                     
         ├──hmdb51_json  
              ├──hmdb51_1.json
              ├──hmdb51_2.json
              |——hmdb51_3.json 
            
   ```
  > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
## 获取预训练模型
下载预训练模型r3d18_K_200ep.pth [here](https://drive.google.com/open?id=1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4)
并放在数据集目录下。
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

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --video_path                        //数据集路径
   --annotation_path                   //标签路径
   --result_path                       //结果路径
   --dataset                           //数据集名称      
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --learning_rate                     //初始学习率，默认：0.01
   --model_depth                       //模型深度
   --n_threads                         //线程
   --amp                               //是否使用混合精度
   --loss_scale_value                  //混合精度lossscale大小
   --opt_level                         //混合精度类型
   --device_list                       //可用设备
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | loss_scaler |
| ------- | ----- | ---: | ------ | -------: |
| 1p-NPU(1.5) | -     |  519.11 | 1      | dynamic |
| 8p-NPU(1.5)  | 0.5895| 2836.99 | 200 |  dynamic |
| 1p-NPU(1.8) | -      | 773 | 1    |  dynamic |
| 8p-NPU(1.8)  | 0.5986 | 5321 | 200    |  dynamic |

# 版本说明

## 变更

2022.08.29：更新Torch1.8版本，重新发布。

2022.07.08：首次发布。

## 已知问题


无。











