# Nonlocal模型-推理指导


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

Nonlocal模型的作用即为了有效捕捉序列中各个元素间的依赖关系。在这里，所谓的序列可以是单幅图像的不同位置（即空间序列），也可以是视频中的不同帧（即时间序列），还可以是视频中不同帧的不同位置（即时空序列）。



- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmaction2
  branch=master
  commit_id=92e5517f1b3cbf937078d66c0dc5c4ba7abf7a08
  model_name=Nonlocal
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout  {branch}            # 切换到对应分支
  git reset --hard ｛commit_id｝    # 代码设置到对应的commit_id
  cd ｛code_path｝                  # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换 
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 数据类型  | 大小                          | 数据排布格式  |
  | -------- | -------- | ------------------------------| ------------ |
  | video    | FLOAT32  | batchsize x 3 x 8 x 224 x 224 | ND           |


- 输出数据

  | 输出数据  | 数据类型  | 大小             | 数据排布格式  |
  | -------- | -------- |----------------- | ------------ |
  | class    | FLOAT32  | batchsize x 400  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| Pytorch                                                      | 1.8.0   | -                                                  
| CANN                                                         | 6.0.0   | -                                                            |
| Python                                                       | 3.7.5   | -                                                            | 
| 操作系统                                                      | Ubuntu 18.04   | -                                                            |                     
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

  **表 2**  版本依赖表

| 依赖                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| onnx                                                         | 1.7.0   | -                                                  
| torchvision                                                  | 0.8.0   | -                                                  
| numpy                                                        | 1.21.6  | -                                                  
| onnx-simplifier                                              | 0.3.6   | -                                                  
| onnxoptimizer                                                | 0.3.1   | -                                                  
| onnxruntime                                                  | 1.21.1  | -                                                  
| decord                                                       | 0.6.0   | -                                                  
| scipy                                                        | 1.7.0   | -                                                            |
| opencv-contrib-python                                        | 4.5.3.56| -                                                  
| opencv-python                                                | 4.5.3.56|-                                                  
| mmcv-full                                                    | 1.3.9   | -                                                  
| einops                                                       | 0.3.0   | -                                                  
| sympy                                                        | 1.10.1  | -                                                  
| pandas                                                       | 1.3.5   | -                                                  
| torchaudio                                                   | 0.7.0   | -         
| cudatoolkit                                                  | 10.1.0  | -                                                  
| decorator                                                    | 5.1.1   | -                                                  
| attrs                                                        | 22.1.0  | -                                                  
| psutil                                                       | 5.9.2   | -                                                  
| absl-py                                                      | 1.2.0   | -                                                  
| tensorflow                                                   | 2.10.0  | -                                                  
| tqdm                                                         | 4.64.1  | -                                                                            




# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取mmaction源码。
   ```
   git clone https://github.com/open-mmlab/mmaction2.git
   cd mmaction2
   git checkout 92e5517f1b3cbf937078d66c0dc5c4ba7abf7a08
   git am --signoff < ../nonlocal.patch
   pip3 install -r requirements/build.txt
   pip3 install -v -e .
   cd ..  
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持 kinetics400 数据集的验证集（video 格式）。

   下载 kinetics400 验证数据集并进行解压。
   ```
   cd kinetics400
   bash k400_downloader.sh
   bash k400_extractor.sh
   ```

   根据 val.csv 文件生成 txt 格式的标注文件。
   ```
   python3 generate_val_txt.py --dataset_root ./ --val_csv k400/annotations/val.csv --video_file k400/val --val_txt val.txt
   cd ..
   ```

   - 参数说明：

      - --dataset_root：数据集的下载路径

      - --val_csv：下载的 val 数据集信息文件

      - --video_file：下载的 val 数据集文件夹

      - --val_txt：生成的 txt 文件名称
   
   > **说明：** 
   > 请根据数据集的实际路径进行相应修改。

   标注文件内容格式如下所示：
   ```
   video_0.mp4 label_0
   video_1.mp4 label_1
   video_2.mp4 label_2
   ...
   video_N.mp4 label_N
   ``` 
    
   若已经预先准备好数据集和文件列表，则需在 mmaction2 文件夹中的相应位置处，软链接到已有文件。
   ```
   mkdir -p ./mmaction2/data/kinetics400
   ln -s kinetics400/k400/val mmaction2/data/kinetics400/videos_val
   ln -s kinetics400/val.txt mmaction2/data/kinetics400/kinetics400_val_list_videos.txt
   ```
   > **说明：** 
   > 请用实际路径替换 kinetics400/k400/val 与 kinetics400/val.txt 。

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集（video 格式）进行数据增强后，转换为模型输入（二进制文件）的数据。

   该部分前处理代码的运行，在用于测试模型精度的脚本 test/eval_acc_perf.sh 中，具体如下所示：
   ```
   python3 tsm_k400_preprocess.py --config mmaction2/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb.py --batch_size 1 --data_root mmaction2/data/kinetics400/videos_val/ --ann_file mmaction2/data/kinetics400/kinetics400_val_list_videos.txt --name out_bin_1
   ```
   
   - 参数说明

      - --config：模型配置文件

      - --batch_size：批大小，即1次迭代所使用的样本量

      - --data_root：数据集路径

      - --ann_file：标注文件路径

      - --name：预处理后的数据文件的相对路径

   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“out_bin_1”二进制文件夹和“k400.info”文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从[源码包](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md)中获取权重文件：“tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth”。

   2. 导出onnx文件。

      1. 使用“tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth”导出onnx文件。

         运行“pytorch2onnx.py”脚本。

         ```
         python3 pytorch2onnx.py \
         mmaction2/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb.py \
         ./tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth \
         --output-file=tsm_nl_1.onnx --softmax --verify --show \
         --shape 1 8 3 224 224
         ```

         获得“tsm_nl.onnx”文件。不同的bs修改output-file与shape即可

         + 参数说明
           + mmaction2/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb.py：使用的开源代码文件路径
           + ./tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth：权重文件名称
           + --output-file：输出文件名称
           + --shape：图片参数


      2. 优化ONNX文件。

         ```
         python3 -m onnxsim --input-shape="1,8,3,224,224" tsm_nl_1.onnx tsm_nl_bs1.onnx
         ```

         获得“tsm_nl_bs1.onnx”文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
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
         atc --model=tsm_nl_bs1.onnx \
         --framework=5 --output=tsm_nl_bs1 \
         --input_format=ND --input_shape="video:1,8,3,224,224" \
         --log=debug --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：ONNX模型文件
           -   --framework：5代表ONNX模型
           -   --output：输出的OM模型
           -   --input\_format：输入数据的格式
           -   --input\_shape：输入数据的shape
           -   --log：日志级别
           -   --soc\_version：处理器型号

         运行成功后生成“tsm_nl_bs1.om”模型文件。不同的bs修改model、output与input_shape即可。



2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   2.  执行推理。

         ```
         python3 -m ais_bench --model ./tsm_nl_bs1.om --input ./mmaction2/data/kinetics400/out_bin_1/ --output ./out/out_1/ --outfmt TXT --batchsize 1  
         ```

         - 参数说明：
           -  input：预处理文件路径
           -  model：om文件路径
           -  outfmt：输出类型

         推理后的输出默认在当前目录./out/out_1下。推理之后将out/out_1/xxx/sumary.json删除。

         > **说明：** 
         > 执行ais-bench工具请选择与运行环境架构相同的命令。

   3.  精度验证。

         调用脚本与数据集标签kinetics400_val_list_videos.txt比对，可以获得Accuracy数据。

         ```
         python3 tsm_k400_postprocess.py --result_path ./out/out_1/xxxx_xx_xx-xx_xx_xx --info_path ./mmaction2/data/kinetics400/k400.info
         ```
         
         - 参数说明：
            - result_path：生成推理结果所在路径 
            - info_path：标签数据
   
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model tsm_nl_bs1.om --loop 1000 --batchsize 1
      ```
    

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能和精度参考下列数据。

|      | 310     | 310P    | T4       | 310P/310| 310P/T4|
|------|---------|---------|----------|---------|--------|
| bs1  | 53.8752 | 97.6916 | 62.1658  | 1.8132  | 1.5714 |
| bs4  | 56.076  | 78.2537 | 67.59371 | 1.3954  | 1.1577 |
| bs8  | 53.1328 | 79.8075 | 68.28619 | 1.5020  | 1.1687 |
| bs16 | 52.618  | 78.5289 | 70.01545 | 1.4924  | 1.1215 |
| bs32 | 51.1944 | 66.8704 | 69.8979  | 1.3062  | 0.9566 |
| bs64 | 52.6652 | 73.1997 | 70.4010  | 1.3899  | 1.0397 |
|最优bs| 56.076  | 97.6916 | 70.4010  | 1.7421  | 1.3876 |

|       | TOP1   | TOP5   |
|-------|--------|--------|
|310精度 | 71.61% | 90.26% |
|310P精度| 71.62% | 90.27% |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
