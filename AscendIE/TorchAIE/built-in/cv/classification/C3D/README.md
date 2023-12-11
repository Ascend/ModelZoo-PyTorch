# C3D模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******

  

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

C3D一种简单而有效的方法，用于使用在大规模监督视频数据集上训练的深层三维卷积网络（3D ConvNets）进行时空特征学习。该网络有三个方面的优势：1）与 2D ConvNets 相比，3D ConvNets 更适合于时空特征学习；2）所有层级的 3×3×3 小卷积核心 的均匀架构是 3D ConvNets 中性能最好的架构之一；3）使用简单的线性分类器学习的 特征，即 C3D（卷积 3D），在 4 个不同的基准上优于最先进的方法，并且与其他 2 个基准 上的当前最佳方法相当。另外，特征非常紧凑：仅使用 10 维的 UCF101 数据集的精度达到 52.8％，由于 ConvNets 的快速推理能力，其计算效率也非常高。最后，它们在概念上很简单，易于训练和使用。


- 参考实现：

  ```
  url=https://github.com/openmmlab/mmaction2/blob/master/configs/recognition/c3d
  branch=master
  commit_id=3e9e99ff7413b2b5c105586000dc0cc793ce00b5
  model_name=c3d
  ```
  


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                               | 数据排布格式 |
  | -------- | -------- | ---------------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 10 x 3 x16 x 112 x 112 | NDCTHW       |


- 输出数据

  | 输出数据 | 数据类型 | 大小           | 数据排布格式 |
  | -------- | -------- | -------------- | ------------ |
  | class    | FP32     | batchsize x101 | ND           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本 | 环境准备指导                                                 |
| ------------------------------------------------------------ |--| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | - | -                                                            |
| Python                                                       | 3.9.0 | -                                                            |
| PyTorch                                                      | 2.0.1 | -                                                            |
| Torchvision                                                  | 15.0.2 |                                                              |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \ | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 获取本仓源码。

2. 在同级目录下，获取第三方库mmaction2源码并安装。

   ```
   pip3 install openmim
   pip3 install mmcv-full==1.4.0
   
   git clone https://github.com/open-mmlab/mmaction2.git        # 克隆仓库的代码
   cd mmaction2                                                 # 切换到模型的代码仓目录
   git checkout 3e9e99ff7413b2b5c105586000dc0cc793ce00b5        # 切换到对应分支
   
   pip3 install -r requirements/build.txt
   pip3 install -v -e .
   ```

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   运行下述命令下载UCF101视频数据集并提取RGB原始帧（视频目录：mmaction2/data/ucf101/videos/，提取好的原始帧目录：mmaction2/data/ucf101/rawframes）

   ```
   cd tools/data/ucf101/
   bash download_videos.sh
   bash extract_rgb_frames_opencv.sh
   
   bash download_annotations.sh
   bash generate_rawframes_filelist.sh
   bash generate_videos_filelist.sh
   ```
   
   本仓代码和mmaction2源码的目录结构组织如下：
   
   ```
   C3D
   |-- C3D_postprocess.py
   |-- C3D_preprocess.py
   |-- C3D_pth2onnx.py
   |-- LICENSE
   |-- README.md
   |-- requirements.txt
   ├── mmaction2 #mmaction2的目录结构
       ├── mmaction
       ├── tools
       ├── configs
       ├── data
       │   ├── ucf101 #数据集目录结构
       │   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
       │   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
       │   │   ├── annotations
       │   │   ├── videos
       │   │   │   ├── ApplyEyeMakeup
       │   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi  
       │   │   │   ├── YoYo
       │   │   │   │   ├── v_YoYo_g25_c05.avi
       │   │   ├── rawframes
       │   │   │   ├── ApplyEyeMakeup
       │   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
       │   │   │   │   │   ├── img_00001.jpg
       │   │   │   │   │   ├── img_00002.jpg
       │   │   │   │   │   ├── ...
       │   │   │   │   │   ├── flow_x_00001.jpg
       │   │   │   │   │   ├── flow_x_00002.jpg
       │   │   │   │   │   ├── ...
       │   │   │   │   │   ├── flow_y_00001.jpg
       │   │   │   │   │   ├── flow_y_00002.jpg
       │   │   │   ├── ...
       │   │   │   ├── YoYo
       │   │   │   │   ├── v_YoYo_g01_c01
       │   │   │   │   ├── ...
       │   │   │   │   ├── v_YoYo_g25_c05
   ```
   
2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行C3D_preprocess.py脚本，将原始帧（rawframes）处理为bin文件。

   ```
   cd ${path_to_C3D}/mmaction2
   mkdir prep_datasets
   python3 ../C3D_preprocess.py ./configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py --output_path ./prep_datasets
   ```
   
	参数说明：
	
	- --参数1：配置文件的路径.
	- --output_path：输出文件夹的位置


## 模型推理<a name="section741711594517"></a>



1. 获取权重文件。

    [pth文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/C3D/PTH/C3D.pth)

2. 开始推理验证。

    a.  执行推理，获取精度和性能数据
	```shell
	# 移除异常数据
    bash ../check_rawframes_filelist.sh
    rm -rf prep_datasets/v_PommelHorse_g05*.bin
    
    # 执行推理
    cd ${path_to_C3D}/mmaction2
    python3 ../c3d_sample.py --config ./configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py --checkpoint ../C3D.pth \
    --label_file ./data/ucf101/ucf101_val_split_1_rawframes.txt --prep_data ./prep_datasets --batch_size 1 --device_id 0
    ```
    参数说明：
    
    - --config：模型结构配置文件。
    - --checkpoint：预训练权重。
    - --label_file：标签文件。
    - --prep_data：数据预处理文件路径。
    - --batch_size：模型batch size 默认为1,参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。
    - --output_dirname：推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中。
    - --device_id:推理设别，默认值为0。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，C3D的精度和性能参考下列数据。

| Batch Size   | 数据集 | 精度指标（Top-1） | 310P3性能 |
| ---------------- | ---------- |-------------|---------|
| 1 | UCF101 | 82.24%      | 45.70   |
| 4 | UCF101 | 82.24%      | 54.69   | 
| 8 | UCF101 | 82.24%       | 56.26   | 
| 16 | UCF101 | 82.24%      | 55.99   | 
| 32         | UCF101 |82.24%      | 52.31   |
