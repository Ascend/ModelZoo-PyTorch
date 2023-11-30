
# 3D_ResNet for PyTorch

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


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取Hmdb51数据集。

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
- 下载预训练模型r3d18_K_200ep.pth，并放在数据集一级目录下，目录结构参考如下所示。
  ```
  ├── data
       ├──hmdb51_jpg
          ...
       ├──hmdb51_json
       ├──r3d18_K_200ep.pth
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --video_path                        //数据集路径
   --annotation_path                   //标签路径
   --result_path                       //结果路径
   --dataset                           //数据集名称      
   --batch-size                        //训练批次大小
   --learning_rate                     //初始学习率，默认：0.01
   --model_depth                       //模型深度
   --n_threads                         //线程
   --loss_scale_value                  //混合精度lossscale大小
   --opt_level                         //混合精度类型
   --device_list                       //训练卡设置
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type | Torch_Version |
| :-----: | :---: | :--: | :----: | :------: | :------: |
| 1p-NPU | -      | 631 | 1    |  O2 | 1.8 |
| 8p-NPU | 0.598 | 5436.46 | 200    |  O2 | 1.8 |

# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

2022.07.08：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md