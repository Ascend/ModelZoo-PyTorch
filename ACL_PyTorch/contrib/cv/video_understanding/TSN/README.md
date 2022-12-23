# TSN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

TSN是一个经典的动作识别网络，在时间结构建模方面，采用稀疏时间采样策略，因为密集时间采样会有大量的冗余相似帧。然后提出可video-level的框架，在长视频序列中提取短片段，同时样本在时间维度均匀分布，由此采用segment结构来聚合采样片段的信息。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmaction2
  branch=master
  commit_id=9ab8c2af52c561e5c789ccaf7b62f4b7679c103c
  model_name=TSN
  ```
  


  通过Git获取对应commit\_id的代码方法如下：

  ```shell
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                           | 数据排布格式 |
  | -------- | -------- | ------------------------------ | ------------ |
  | input    | RGB_FP32 | batchsize x 75 x 3 x 256 x 256 | NTCHW        |


- 输出数据

  | 输出数据 | 大小            | 数据类型 | 数据排布格式 |
  | -------- | --------------- | -------- | ------------ |
  | output1  | batchsize x 101 | FLOAT32  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

​          第三方库：

```
absl-py               0.13.0
aclruntime            0.0.1
addict                2.4.0
apex                  0.1+ascend
attrs                 21.2.0
auto-tune             0.1.0
certifi               2021.5.30
commonmark            0.9.1
cycler                0.11.0
decorator             5.0.9
fonttools             4.36.0
future                0.18.2
hccl                  0.1.0
hccl-parser           0.1
imageio               2.9.0
kiwisolver            1.4.4
matplotlib            3.5.3   
mmcv-full             1.3.9
mpmath                1.2.1
msadvisor             1.0.0
numpy                 1.21.2
onnx                  1.12.0
onnx-simplifier       0.4.7
op-gen                0.1
op-test-frame         0.1
opc-tool              0.1.0
opencv-contrib-python 4.6.0.66
opencv-python         4.6.0.66
packaging             21.3
pandas                1.3.5
Pillow                9.2.0
pip                   21.2.2
protobuf              3.20.1
psutil                5.8.0
Pygments              2.13.0
pyparsing             3.0.9
python-dateutil       2.8.2
pytz                  2022.2
PyYAML                6.0
rich                  12.5.1
schedule-search       0.0.1
six                   1.16.0
some-package          0.1
sympy                 1.10.1
te                    0.4.0
topi                  0.4.0
torch                 1.5.0+ascend.post3
torchvision           0.6.0
tqdm                  4.64.0
typing_extensions     4.3.0
wheel                 0.37.0
xarray                0.18.2
yapf                  0.32.0
```



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖

   ```shell
   pip3 install -r requirment.txt
   ```
   
   
   
1. 获取开源代码仓，并安装自带依赖

   在已下载的源码包根目录下，执行如下命令。
   
   ```shell
   git clone https://github.com/open-mmlab/mmaction2.git
   cd mmaction2
   git checkout 9ab8c2af52c561e5c789ccaf7b62f4b7679c103c
   pip install -r requirements/build.txt
   pip install -v -e .
   cd ..
   ```
   
   

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用UCF101数据集和对应annotations标签文件，详细介绍参见[UCF101官方下载地址](https://www.crcv.ucf.edu/data/UCF101.php 

   可使用开源代码仓的脚本下载和组织数据，具体步骤如下：

   1. 在已下载的源码包根目录下，执行如下命令：
      ```
      cd ./mmaction2/tools/data/ucf101
      bash download_annotations.sh  # 下载ucf101标签注释文件
      bash download_videos.sh       # 下载ucf101原始视频文件
      cd ../../../data/ucf101/
      unrar e ucf101.rar            # 默认下载的*.rar压缩包，需要联系工作人员安装unrar帮忙解压
      ```
      解压后得到的数据目录结构，仅供参考：
      ```
      ├──ucf101
          ├──videos
          ├──annotations
                ├── classInd.txt
                ├── testlist01.txt
                ├── testlist02.txt
                ├── testlist03.txt
                ├── trainlist01.txt
                ├── trainlist02.txt
                └── trainlist03.txt
                ...
      ```

   2. 在已下载的源码包根目录下，执行如下命令提取视频帧并生成对应的数据集文件列表：

      ```
      cd ./mmaction2/tools/data/ucf101    # 通过开源代码仓提供的数据处理脚本处理数据
      bash extract_rgb_frames_opencv.sh   # 通过opencv从视频中提取rgb帧
      bash generate_videos_filelist.sh    # 生成视频数据集list文件
      bash generate_rawframes_filelist.sh # 生成帧数据集list文件 
      ```

      

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   在已下载的源码包根目录下，执行tsn_ucf101_preprocess.py脚本，完成预处理。

   ```
   python3.7 tsn_ucf101_preprocess.py --batch_size 1 --data_root ./mmaction2/data/ucf101 --name out_bin_1
   ```

   - 参数说明：

     -   --batch_size：处理批次。

   
     -   --data_root：原始数据根目录路径。
   

     -   --name：输出的二进制文件（.bin）所在路径。

   预处理后会生成标签文件ucf101_1.info，并在out_bin_1文件夹生成二进制文件作为模型的输入

   预处理后的数据目录结构，仅供参考：
   
   ```
   -- ucf101
       |-- annotations
       |-- out_bin_1
       |-- rawframes
       |-- ucf101_1.info
       |-- ucf101_train_split_1_rawframes.txt
       |-- ucf101_train_split_1_videos.txt
       |-- ucf101_train_split_2_rawframes.txt
       |-- ucf101_train_split_2_videos.txt
       |-- ucf101_train_split_3_rawframes.txt
       |-- ucf101_train_split_3_videos.txt
       |-- ucf101_val_split_1_rawframes.txt
       |-- ucf101_val_split_1_videos.txt
       |-- ucf101_val_split_2_rawframes.txt
       |-- ucf101_val_split_2_videos.txt
       |-- ucf101_val_split_3_rawframes.txt
       |-- ucf101_val_split_3_videos.txt
       `-- videos
   ```

   
   
   


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      在已下载的源码包根目录下，执行

      ```
      wget -c https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth
      ```

   2. 导出onnx文件。

      使用pytorch2onnx.py导出onnx文件。

      在已下载的源码包根目录下，运行pytorch2onnx.py脚本。

      ```
      python3.7 pytorch2onnx.py mmaction2/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py ./tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth --output-file=./tsn.onnx --verify
      ```

      - 参数说明：<br>

        - ${config-path}：位置参数1，配置文件路径<br>

        - ${checkpotin-path}：位置参数2，checkpoints文件路径<br>
        
         - --output-file：转换后的.onnx文件输出路径
        
        
         - --verify：是否对照pytorch输出验证onnx模型输出

   2. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
         
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。


      2. 执行命令查看芯片名称。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
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

         在已下载的源码包根目录下，执行

         ```
         atc --framework=5 --model=tsn.onnx --output=tsn_bs1_710 --input_format=NCDHW --input_shape="image:1,75,3,256,256" --log=debug --soc_version=Ascend310P3
         ```

         - 参数说明：

           - --model：为ONNX模型文件。

           - --framework：5代表ONNX模型。

           - --output：输出的OM模型。

           - --inputformat：输入数据的格式。

           - --input\_shape：输入数据的shape。"image:1,75,3,256,256"中的1代表batchsize大小。

           - --log：日志级别。

           - --soc\_version：处理器型号。



         运行成功后生成<u>***tsn_bs1_710.om***</u>模型文件。

         

2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

      在已下载的源码包根目录下, 可参考下列命令安装：

      ```
      export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest  # 指定CANN包的安装路径
      
      git clone https://gitee.com/ascend/tools.git  # 获取源码
      cd tools/ais-bench_workload/tool/ais_infer/  # 打包
      
      pip3  wheel ./backend/ -v
      pip3  wheel ./ -v
      
      pip3 install --force-reinstall ./aclruntime-{version}-cp37-cp37m-linux_xxx.whl  # 安装
      pip3 install --force-reinstall ./ais_bench-{version}-py3-none-any.whl
      ```

   2. 执行推理。

      **注：该模型因为太大，只支持bs1、4、8，bs1能跑数据推理，其他bs因为推理输出太大，故使用纯推理。**

      1. 性能推理

         在已下载的源码包根目录下，执行如下命令：

         ```
         cd tools/ais-bench_workload/tool/ais_infer/  # 移动至ais_bench推理工具所在目录 
         mkdir out_tmp  # 创建一个存储纯推理结果的临时目录
         python3 -m ais_bench --model ../../../../tsn_bs1_710.om --output ./tmp_out --batchsize 1 --outfmt TXT --loop 5
         ```

         - 参数说明：

           -  --model：OM文件路径。


           -  --output：推理结果的保存目录。


           -  --batchsize：批大小。


           -  --outfmt: 输出数据格式。


           -  --loop：推理次数，可选参数，默认1，profiler为true时，推荐为1。


          **说明：** 

         > 执行ais_bench工具请选择与运 行环境架构相同的命令。参数详情请参见：https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer

      2. 精度测试

         在已下载的源码包根目录下，执行如下命令：

         ```
         mkdir result  # 创建一个存储真实数据推理结果的临时目录
         cd tools/ais-bench_workload/tool/ais_infer/  # 移动至ais_bench推理工具所在目录 
         python3-m ais_bench --model ../../../../tsn_bs1_710.om --input ../../../../mmaction2/data/ucf101/out_bin_1/  --output ../../../../result/ --batchsize 1 --outfmt TXT 
         ```

         - 参数说明：

             - --model：OM文件路径。

    
             - --input：预处理bin文件所在目录的路径，纯推理时不需要指定。


             - --output：推理结果的保存目录。（使用真实数据推理时，建议在源码包根目录下创建一个文件夹存储输出数据，方便调用脚本测试精度）


             - --batchsize：批大小。


             - --outfmt: 输出数据格式

  

         调用脚本与数据集标签ucf101_1.info比对，可以获得Accuracy数据。

         在已下载的源码包根目录下，执行如下命令：

         ```
         python tsn_ucf101_postprocess.py --result_path result/2022_09_02-06_42_04/ --info_path ./mmaction2/data/ucf101/ucf101_1.info --batch_size 1
         ```

         - 参数说明：

            -    --result_path：aisinfer推理结果所在路径。


            -    --info_path：数据集标签文件。执行“tsn_ucf101_preprocess.py”脚本时生成的。


            -    --batch_size：批大小。






 ## 模型推理性能&精度   

调用ACL接口推理计算，性能参考下列数据。  

| 芯片型号 | Batch Size   | 数据集 | 精度 （top1 acc） | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| Ascend310  | 1 | UCF101 | 82.84% | 22.6680fps |
| Ascend310  | 4 | UCF101 |        | 21.9128fps |
| Ascend310  | 8 | UCF101 |        | 22.6616fps |
| Ascend310P | 1 | UCF101 | 82.84% | 28.0875fps |
| Ascend310P | 4 | UCF101 |        | 28.4336fps |
| Ascend310P | 8 | UCF101 |        | 28.7904fps |
| T4         | 1 | UCF101 |        | 21.6811fps |
| T4         | 4 | UCF101 |        | 22.4574fps |
| T4         | 8 | UCF101 |        | 22.7878fps |

310最优batch size为1，310P最优batch size为8，T4最优batch size为8。
310P最优 / 310最优 = 1.27，310P最优 / T4最优 = 1.26。

