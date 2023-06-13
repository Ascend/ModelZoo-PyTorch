# R(2+1)D模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

R(2+1)D是应用于视频理解领域的时空卷积模块，它明确地将3d卷积分解为两个单独的和连续的操作，一个2d空间卷积和一个1d时间卷积。。与使用相同数量参数的全三维卷积的网络相比，这有效地使非线性数量加倍，从而使模型能够表示更复杂的函数。同时，分解促进了优化，在实践中产生了较低的训练损失和较低的测试损失。


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

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |


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

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```

2. 获取开源代码仓。

   

1. 下载mmcv-full。

   ```
   git clone https://github.com/open-mmlab/mmcv -b master 
   cd mmcv
   git reset --hard 6cb534b775b7502f0dcc59331236e619d3ae5b9f
   MMCV_WITH_OPS=1 pip3.7 install -e .
   cd ..
   ```

2. 下载mmaction2代码仓。

   ```
   git clone https://github.com/open-mmlab/mmaction2 -b master
   cd mmaction2 
   git reset --hard acce52d21a2545d9351b1060853c3bcd171b7158
   python3.7 setup.py develop
   cd ..
   ```

3. 在“mmaction2/tools/deployment/”目录下的“pytorch2onnx.py”中的torch.onnx.export添加一个参数：dynamic_axes={'0':{0:'-1'}}。

   执行**vi mmaction2/tools/deployment/pytorch2onnx.py**，添加 参数后执行**wq**保存。

   ```
   torch.onnx.export(
           model,
           input_tensor,
           output_file,
           export_params=True,
           keep_initializers_as_inputs=True,
           verbose=show,
           opset_version=opset_version,
           dynamic_axes={'0':{0:'-1'}})
   ```
   
4. 移动数据集配置文件至“mmaction2/configs/recognition/r2plus1d”

   ```
   mv  r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py mmaction2/configs/recognition/r2plus1d
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   模型支持UCF-101 3783个视频的验证集，请用户自行获取UCF-101数据集，上传数据集到服务器任意目录并解压（如：/root/datasets）。

      数据目录结构请参考：

      ```
      ├──ucf101
               ├──ApplyEyeMakeup
                .......
               ├──YoYo
      ```

      照下方命令对视频数据进行处理，处理后的数据格式为从视频帧中提取的jpg图片。

   创建路径：

   ```
   mkdir -p ./data/ucf101/videos  #在模型文件夹内创建数据及路径
   cp -r /root/datasets/ucf101/*  ./data/ucf101/videos  #文件夹下的视频文件夹复制到videos下
   ```

   将视频处理为图片：

   ```
   python3.7 ./mmaction2/tools/data/build_rawframes.py ./data/ucf101/videos/ ./data/ucf101/rawframes/ --task rgb --level 2 --ext avi --use-opencv
   ```

   下载标签文件：

   ```
   wget https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
   mkdir ./data/ucf101/annotations  
   unzip -j UCF101TrainTestSplits-RecognitionTask.zip -d ./data/ucf101/annotations  
   ```

   生成数据文件：

   ```
   python3.7 ./mmaction2/tools/data/build_file_list.py ucf101 data/ucf101/rawframes/ --level 2 --format rawframes --shuffle
   ```

   

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行“r2plus1d_preprocess.py”脚本，完成预处理。

      ```
      python3.7 r2plus1d_preprocess.py --config=./mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py --bts=1 -- output_path=./predata_bts1/
      ```
   
    -  参数说明：

       -   --config：数据集UCF-101的配置文件。
       -   --bts：batch_size。
       -   --output_path：生成的bin文件的位置。


   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“predata_bts1”二进制文件夹。

   数据预处理将原始数据集转换为模型输入的数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件：“[best_top1_acc_epoch_35.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/R%282%2B1%29D/PTH/best_top1_acc_epoch_35.pth)”

   2. 导出onnx文件。

      1. 使用“best_top1_acc_epoch_35.pth”导出onnx文件。

         运行“pytorch2onnx.py”脚本。onnx文件与优化的onnx文件为动态

         ```
         python3.7 ./mmaction2/tools/deployment/pytorch2onnx.py ./mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py best_top1_acc_epoch_35.pth --verify  --output-file=r2plus1d.onnx --shape 1 3 3 8 256 256
         ```

         获得“r2plus1d.onnx”文件。

      2. 优化ONNX文件。

         ```
          python3.7 -m onnxsim --input-shape="1,3,3,8,256,256" --dynamic-input-shape r2plus1d.onnx r2plus1d_sim.onnx
         ```

         获得“r2plus1d_sim.onnx”文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
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
   
         例：batchsize=1
   
         ```
         atc --framework=5 --model=./r2plus1d_sim.onnx --output=r2plus1d_bs1 --input_format=NCHW --input_shape="0:1,3,3,8,256,256" --log=debug --soc_version=Ascend${chip_name}
         ```
   
         例：batchsize=16
   
          ```
          atc --framework=5 --model=./r2plus1d_sim.onnx --output=r2plus1d_bs16 --input_format=NCHW --input_shape="0:16,3,3,8,256,256" --log=debug --soc_version=Ascend${chip_name}
          ```
   
         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
   
           运行成功后生成“r2plus1d_bs1.om”模型文件。



2. 开始推理验证。

   

​		a.  安装ais_bench推理工具。
         请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

​		b.  执行推理。

    ```
    mkdir predata_bts1_om_out
    python3 -m ais_bench --model /home/HwHiAiUser/r2plus1d_bs1.om --input  /home/HwHiAiUser/predata_bts1  --batchsize 1 --output "./predata_bts1_om_out" --outfmt TXT
    ```
    
    -   参数说明：
    
        --  model：输入的om文件
        --  input：输入的bin数据文件。
        --  outfmt：输出数据的格式
    	...


​		c.  精度验证。			

    用“r2plus1d_postprocess.py”脚本将推理结果处理成json文件。
    
    ```
    cd R(2+1)D
    python3.7 r2plus1d_postprocess.py --result_path=./predata_bts1_om_out > result_bs1.json
    ```
    
    --result_path：标签数据路径。
    
    “result_bs1.json”：精度数据文件。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | TOP1   | TOP5   |
| -------- | ------ | ------ |
| 310精度  | 0.8892 | 0.9741 |
| 710精度  | 0.8923 | 0.9745 |

|        | 310     | 310p    | t4      | 310p/310 | 310p/t4 |
| ------ | ------- | ------- | ------- | -------- | ------- |
| bs1    | 44.9012 | 60.6969 | 38.2989 | 1.3517   | 1.5848  |
| bs4    | 49.2828 | 80.4857 | 38.8568 | 1.6331   | 2.2446  |
| bs8    | 48.0392 | 83.7505 | 39.5354 | 1.7433   | 2.1183  |
| bs16   | 49.0324 | 84.0091 | 40.3182 | 1.7133   | 2.0836  |
| bs32   | 52.6588 | 84.7777 | 31.6976 | 1.6099   | 2.6745  |
| 最优bs | 52.6588 | 84.7777 | 40.3182 | 1.6099   | 2.1027  |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md