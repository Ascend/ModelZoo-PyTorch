# 3DMPPE-ROOTNET模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

三维人体姿态估计的目标是在三维空间中定位单个或多个人体的语义关键点，使得模型能够理解人类行为，进行人机交互。3DMPPE-ROOTNET是一个三维多人姿态估计的通用框架，在几个公开可用的3D单人和多人姿态估计数据集上都取得了最好的水平。


- 参考论文：[Gyeongsik Moon, Ju Yong Chang, Kyoung Mu Lee. Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image.(2019)](https://arxiv.org/abs/1907.11346)
- 参考实现：

  ```
  url=https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
  branch=master
  commit_id=a199d50be5b0a9ba348679ad4d010130535a631d
  ```



  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 大小                        | 数据类型    | 数据排布格式 |
  | -------- | ----------------------------| ----------- | ------------ |
  | input1    | batchsize x 3 x 224 x 224   | RGB         | NCHW         |
  | input2    | batchsize x 1               | FLOAT32     | ND           |


- 输出数据

  | 输出数据 | 大小           | 数据类型 | 数据排布格式 |
  | -------- | -------------- | -------- | ------------ |
  | output  |  batchsize x 3 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.16（NPU驱动固件版本为5.1.RC2）  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | - 
  | Python                                                        | 3.7.5 | - 



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   <!--```-->
   <!--https://www.hiascend.com/zh/software/modelzoo/models/detail/1/c7f19abfe57146bd8ec494c0b377517c-->
   <!--```-->
    源码目录结构：
    ``` 
    ├── 3DMPPE-ROOTNET_postprocess.py              //模型后处理脚本  
    ├── 3DMPPE-ROOTNET_preprocess.py               //模型前处理脚本 
    ├── 3DMPPE-ROOTNET_pth2onnx.py                 //用于转换pth文件到onnx文件  
    ├── 3DMPPE-ROOTNET.patch                        //用于修改开源模型代码的文件    
    ├── modelzoo_level.txt                          //模型精度性能结果
    ├── requirements.txt                            //依赖库和版本号
    ├── LICENSE                                     //Apache LICENCE                            
    ├── README.md                                   //模型离线推理说明README
    ```
2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。
   ```
   git clone https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE.git 
   cd 3DMPPE_ROOTNET_RELEASE  
   patch -p1 < ../3DMPPE_ROOTNET.patch 
   cd ..
   ```
3. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型基于MuPoTS数据集推理，MuPoTS是一个多人三维人体姿态估计数据集，由超过8000帧来自20个真实场景，最多3个对象组成。
   请用户自行获取MuPoTS数据集并解压，自行获取数据集对应的annotation文件。存放到新建的MuPoTS文件夹内。上传MuPoTS文件夹到服务器任意目录并解压（如：/root/datasets/）
   数据目录结构请参考：
    ```
    ├──MuPoTS
        ├──MultiPersonTestSet
        ├──MuPoTS-3D.json
    ```
   

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。
   执行3DMPPE-ROOTNET_preprocess.py脚本，完成预处理。

   ```
   python 3DMPPE-ROOTNET_preprocess.py --img_path=./MuPoTS/MultiPersonTestSet --ann_path=./MuPoTS/MuPoTS-3D.json --save_path_image=data_image_bs1 --save_path_cam=data_cam_bs1 --inference_batch_siz=1
   ```
    
   - 参数说明：
      -    --img_path：原始数据验证集所在路径。
      -    --save_path_image：bin文件保存路径。
      -    --save_path_cam：bin文件保存路径。
      -    --inference_batch_siz：处理批次数。

   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“data_cam_bs1”和“data_image_bs1”二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件 [snapshot_6.pth.tar](https://pan.baidu.com/s/15gzQpHGflKB9QcoEZ6XbYQ)。

   2. 导出onnx文件。

        使用“snapshot_6.pth.tar”导出onnx文件。

        运行“3DMPPE-ROOTNET_pth2onnx.py”脚本。

         ```
         python 3DMPPE-ROOTNET_pth2onnx.py snapshot_6.pth.tar 3DMPPE-ROOTNET.onnx
         ```
         - 参数说明：

           -   --snapshot_6.pth.tarl：为权重文件路径。
           -   --3DMPPE-ROOTNET.onnx：onnx文件的输出路径。

         获得“3DMPPE-ROOTNET.onnx”文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称(${chip_name})。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 17.6         57                0    / 0              |
         | 0       0         | 0000:3B:00.0    | 0            936 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。
         ```
         atc --framework=5 --model=3DMPPE-ROOTNET.onnx --output=3DMPPE-ROOTNET_bs1 --input_format=NCHW --input_shape="image:1,3,256,256;cam_param:1,1" --log=error --soc_version=${chip_name}
         ```
         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

          运行成功后生成<u>“3DMPPE-ROOTNET _bs1.om”</u>模型文件。



2. 开始推理验证。

   1.  使用ais_bench工具进行推理。ais_bench工具获取及使用方式请点击查看 [ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

   2.  创建输出的文件夹。

       ```
       mkdir out_bs1
       ```

      
   3.  执行推理。

       执行命令
       ```
       python -m ais_bench --device 0 --batchsize 1 --model 3DMPPE-ROOTNET_bs1.om --input "data_image_bs1,data_cam_bs1" --output out_bs1
       ```
       -   参数说明：
      
            -   --model ：输入的om文件。
            -   --input：输入的bin数据文件。
            -   --device：NPU设备编号。
            -   --output: 模型推理结果。
            -   --batchsize : 批大小。
       
       推理结果保存在out_bs1下面，并且也会输出性能数据。


   4.  精度验证。

       运行后处理脚本“3DMPPE-ROOTNET _postprocess.py”。

       ```
       python 3DMPPE-ROOTNET_postprocess.py --img_path=MuPoTS/MultiPersonTestSet --ann_path=MuPoTS/MuPoTS-3D.json --input_path=out_bs1 --result_file=result_bs1
       ```
       -   参数说明：
            -   --img_path ：数据集路径。
            -   --ann_path ：数据集标签路径。
            -   --input_path ：模型推理结果。
            -   --result_file：推理结果的accuracy数据。


   5. 性能验证。

       可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

       ```
       python -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
       ```
       - 参数说明：
            - --model：om模型的路径
            - --loop: 推理次数
            - --batchsize：数据集batch_size的大小


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


| 芯片型号 | Batch Size   |  数据集    |   精度     |     性能        |
| ---------| ------------ | ---------- | ---------- | --------------- |
|   310P   |       1      |  MuPoTS    |  0.3181    |      900.384    |
|   310P   |       4      |  MuPoTS    |  0.3181    |      1565.36    |
|   310P   |       8      |  MuPoTS    |  0.3181    |      1420.831    |
|   310P   |       16      |  MuPoTS    |  0.3179    |      1311.963    |
|   310P   |       32     |  MuPoTS    |  0.3175    |      1300.503    |
|   310P   |       64      |  MuPoTS    |  0.3175    |      1009.593    |