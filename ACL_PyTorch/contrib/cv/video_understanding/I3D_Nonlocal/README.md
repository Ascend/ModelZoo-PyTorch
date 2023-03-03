# I3D_Nonlocal模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)   
 
  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

I3D是一种新的基于2D ConvNet 膨胀的双流膨胀3D ConvNet (I3D)。一个I3D网络在RGB输入上训练，另一个在流输入上训练，这些输入携带优化的、平滑的流信息。 模型分别训练了这两个网络，并在测试时将它们的预测进行平均后输出。深度图像分类ConvNets的过滤器和池化内核从2D被扩展为3D，从而可以从视频中学习效果良好的时空特征提取器并改善ImageNet的架构设计，甚至是它们的参数。


- 参考论文：[Carreira, Joao, and Andrew Zisserman. "Quo vadis, action recognition? a new model and the kinetics dataset." proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.](http://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)

- 参考实现

  ```
  url=https://github.com/open-mmlab/mmaction2
  branch=master
  commit_id=dbf5d59fa592818325285b786a0eab8031d9bc80
  ```
  
  适配昇腾 AI 处理器的实现： 
  
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch
  tag=v.0.4.0
  code_path=ACL_PyTorch/contrib/cv/video_understanding
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

  | 输入数据 | 数据类型 | 大小                                | 数据排布格式 |
  | -------- | -------- | ----------------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 10 x 3 x 32 x 256 x 256 | NCHW         |


- 输出数据

  | 输出数据 | 大小            | 数据类型 | 数据排布格式 |
  | -------- | --------------- | -------- | ------------ |
  | output  | 10 x 400 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>


- 该模型需要以下插件和驱动。

  **表 1**  版本配套表

| 配套                                                        | 版本    | 环境准备指导                                                 |
| ----------------------------------------------------------- | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                        | 5.1.RC2 | -                                                            |
| PyTorch                                                     | 1.5.0   | -                                                            |


- 该模型需要以下依赖。

  **表 2**  依赖列表

| 配套                                                        | 版本    | 环境准备指导                                                 |
| ----------------------------------------------------------- | ------- | ------------------------------------------------------------ |
| Python                                                      | 3.8.13  | -                                                            |
| PyTorch                                                     | 1.5.0   | -                                                            |
|acl               |0.1|-|
|addict            |2.4.0|-|
|auto-tune         |0.1.0|-|
|certifi           |2021.5.30|-|
|commonmark        |0.9.1|-|
|dataclasses       |0.8|-|
|einops            |0.4.1|-|
|flatbuffers       |2.0.7|-|
|future            |0.18.2|-|
|hccl              |0.1.0|-|
|hccl-parser       |0.1|-|
|mmaction2         |0.24.1|-|
|mmcv              |1.3.9|-|
|mmcv-full         |1.6.1|-|
|msadvisor         |1.0.0|-|
|numpy             |1.19.5|-|
|onnx              |1.9.0|-|
|onnx-simplifier   |0.4.8|-|
|onnxruntime       |1.10.0|-|
|op-gen            |0.1|-|
|op-test-frame     |0.1|-|
|opc-tool          |0.1.0|-|
|opencv-python     |4.5.3.56|-|
|packaging         |21.3|-|
|Pillow            |8.4.0|-|
|pip               |21.2.2|-|
|protobuf          |4.21.0|-|
|Pygments          |2.13.0|-|
|pyparsing         |3.0.9|-|
|PyYAML            |6.0|-|
|rich              |12.5.1|-|
|schedule-search   |0.0.1|-|
|scipy             |1.5.4|-|
|setuptools        |58.0.4|-|
|six               |1.16.0|-|
|te                |0.4.0|-|
|topi              |0.4.0|-|
|torch             |1.5.0|-|
|torchvision       |0.6.0|-|
|typing_extensions |4.1.1|-|
|wheel             |0.37.1|-|
|yapf              |0.32.0|-|





# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 单击“立即下载”，下载源码包。

2. 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。

   ```
   ├── acl_net.py                       //pyacl推理工具代码 
   ├── i3d_inference.py                   //i3d模型推理脚本 
   ├── generate_labels.py                //i3d验证用标签文件生成脚本 
   ├── LICENSE                           //LICENSE
   ├── requirements.txt           
   ├── modelzoo_level.txt   
    |──README.md
    |──build_rawframes.sh
    |──i3d_infer.sh
    |──i3d_onnx2om.sh
    |──i3d_pth2onnx.sh    
   ```


3. 获取开源代码仓。

   在已下载的源码包根目录下，执行如下命令。
   
   ```
   git clone https://github.com/open-mmlab/mmaction2.git
   cd mmaction2
   git checkout dbf5d59fa592818325285b786a0eab8031d9bc80
   ```
   
4. 放置脚本文件。

   1. 首先创建目录

      ```
      mkdir ./data/kinetics400
      ```

   2. 将“acl_net.py”，“env.sh”放置在“mmaction2”目录下。

      ```
      mv ../acl_net.py ./ &&  mv ../env.sh ./
      ```

   3. 将“i3d_inference.py”放置在“mmaction2/tools”目录下。

      ```
      mv ../i3d_inference.py ./tools 
      ```

   4. 将“generate_lables.py”放置在“mmaction2/data/kinetics400”目录下。

      ```
      mv ../generate_lables.py ./data/kinetics400 
      ```

   

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型支持kinetics400验证集，请用户自行获取[验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB)。该数据集由视频组成，预处理后，将在“rawframes_val”目录下根据视频名称生成对应的目录，每个目录中是对应视频抽出来的帧。请将数据集解压并放置在“mmaction2/data/kinetics400”目录下，为“kinetics_400_val_320.tar”，并重命名为“videos_val”。

   目录结构如下：

   ```shell
   ├── kinetics400
         ├── videos_val
         ├── kinetics_400_val_320.tar
         ├── generate_labels.py
   ```

2. 数据预处理。

   对videos_val中的所有视频进行抽帧处理，并将结果放置在“data/kinetics400/rawframes_val”目录下。本脚本采用Opencv对mp4格式的视频，采用4线程抽取256*256大小的RGB帧，输出格式为jpg。

   ```
   python3 tools/data/build_rawframes.py data/kinetics400/videos_val data/kinetics400/rawframes_val --task rgb --level 1 --num-worker 4 --out-format jpg --ext mp4 --new-width 256 --new-height 256 --use-opencv
   ```

   - 参数说明：
     - --task：提取任务，说明提取帧，光流，还是都提取，选项为 rgb, flow, both。
     - --level：目录层级。1 指单级文件目录，2 指两级文件目录。
     - --num-worker：提取原始帧的线程数。
     - --out-format：提取帧的输出文件类型，如 jpg, h5, png。
     - --ext：视频文件后缀名，如 avi, mp4。
     - --new-width：调整尺寸后，输出图像的宽。
     - --new-height：调整尺寸后，输出图像的高。
     - --use-opencv：是否使用 OpenCV 提取 RGB 帧。

3. 生成验证集标注文件。

   1. 首先，下载kinetics400验证集的标注文件。

      ```
      cd tools/data/kinetics
      bash download_backup_annotations.sh  kinetics400
      cd ../../..
      ```

   2. 进入“data/kinetics400”目录，执行“generate_labels.py”脚本

      ```
      cd data/kinetics400
      python3 generate_labels.py
      cd ../..
      ```
      
        生成推理所需要的标注文件：“kinetics400_val_list_rawframes.txt”。

   


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取[权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/I3D-Nonlocal/PTH/i3d_nl_dot_product_r50.pth)，并新建目录“mmaction2/checkpoints”，将权重文件重命名为“i3d_nl_dot_product_r50.pth”，并保存在“checkpoints”目录下。

   2. 导出onnx文件。

      使用“i3d_nl_dot_product_r50.pth”导出onnx文件。

      运行“pytorch2onnx.py”脚本，获得“i3d_nl_dot.onnx”文件。（本模型只支持bs1）

      ```
      python3 tools/deployment/pytorch2onnx.py configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py checkpoints/i3d_nl_dot_product_r50.pth --shape 1 10 3 32 256 256 --output-file i3d_nl_dot.onnx
      ```
      - 参数说明：
         - --output-file：转换后的.onnx文件输出路径
         - --shape：数据的shape

      > **说明：**
      >
      > 如果上述命令转换的onnx模型测试性能不达标，使用onnx-simplifier进行优化。
      >
      > 1. pip install onnx-simplifier
      > 2. python -m onnxsim --input-shape="1,10,3,32,256,256"  ./././i3d_nl_dot.onnx  ./././i3d_sim.onnx

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量

         ```
         export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
         
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。     

      2. 执行命令查看芯片名称（${chip_name}）

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3(自行替换）
         
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       310P3     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```
         atc --framework=5 --output=./i3d_nl_dot_bs1  --input_format=NCHW  --soc_version=Ascend${chip_name} --model=./i3d_sim.onnx --input_shape="0:1,10,3,32,256,256"
         ```
         - 参数说明：

             - --model：为ONNX模型文件。

             - --framework：5代表ONNX模型。

             - --output：输出的OM模型。

             - --input_format：输入数据的格式。

             - --input_shape：输入数据的shape。

             - --log：日志级别。

             - --soc_version：处理器型号。

         运行成功后生成“i3d_nl_dot_bs1.om”模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   2. 执行推理。
         在已下载的源码包根目录下，执行如下命令：

         ```
         mkdir out_tmp  # 创建一个存储纯推理结果的临时目录
         python3 -m ais_bench --model ./i3d_nl_dot_bs1.om --output ./out_tmp --batchsize 1 --outfmt TXT --loop 5
         ```

         - 参数说明：

           -  --model：OM文件路径。

           - --output：推理结果的保存目录。

           - --batchsize：批大小。

           - --outfmt: 输出数据格式。

           - --loop：推理次数，可选参数，默认1，profiler为true时，推荐为1。
         
   3. 精度验证

      ```
      python tools/i3d_inference.py configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py --eval top_k_accuracy mean_class_accuracy --out result.json -bs 1 --model i3d_nl_dot_bs1.om --device_id 0
      ```
      
      - 参数说明：
        - --eval：精度测试种类
        
        - --out：输出文件名称 
        
        - --bs：batch_size，只能为1
        
        - --device_id：芯片序号（0，1，2，3）
        
          


 ## 模型推理性能&精度   

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集          | 精度 （top1 acc） | 性能         |
| --------- | ---------------- |--------------|---------------|------------|
| Ascend310 | 1 | kinetics400  | 69.99%        | 3.5102 fps |
| Ascend310P | 1 | kinetics400  | 70.07%        | 14.0639 fps |

