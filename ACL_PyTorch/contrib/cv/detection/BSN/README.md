# BSN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

在视频局部，BSN 首先以高概率定位时间边界，然后直接将这些边界组合为提议。 在整体范围内，通过边界敏感提案功能，BSN 通过评估提案是否包含其区域内的动作的置信度来检索提案。

- 参考论文：[Lin, Tianwei, et al. "Bsn: Boundary sensitive network for temporal action proposal generation." Proceedings of the European Conference on Computer Vision (ECCV). 2018.](http://arxiv.org/abs/1806.02964)

参考实现：

```
url=https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch
branch=master
commit_id=e50d12953ec51c128360181afe69db37298f30d2
```

适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/ModelZoo-PyTorch
tag=v.0.4.0
code_path=ACL_PyTorch/contrib/cv/detection
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

  | 输入数据  | 数据类型 | 大小                  | 数据排布格式 |
  | --------- | -------- | --------------------- | ------------ |
  | TEM model | FLOAT32  | batchsize x 400 x 100 | NCHW         |
  | PEM model | FLOAT32  | batchsize x 3 x 100   | ND           |
  
- 输出数据

  | 输出数据  | 数据类型 | 大小                 | 数据排布格式 |
  | --------- | -------- | -------------------- | ------------ |
  | TEM model | FLOAT32  | batchsize x 3 x 100  | ND           |
  | PEM model | FLOAT32  | batchsize x 1000 x 1 | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  
  | 配套                                                         | 版本                | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.83.10              | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1             | -                                                            |
  | Python                                                       | 3.7.5               | -                                                            |
  | PyTorch                                                      | 1.5.0               | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |                     |                                                              |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 单击“立即下载”，下载源码包。

2. 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。

   ```
   ├──test
        ├──eval_acc_perf.sh     //验证精度、性能脚本
        ├──pth2om.sh                //模型转换脚本
        ├──parse.py             //输出保存的性能信息 
   ├── gen_dataset_info.py         //用于获取二进制数据集信息的脚本 
   ├── BSN_eval.py             //测试精度文件 
   ├── BSN_pem_postprocess.py      //pem后处理文件
   ├── BSN_tem_postprocess.py      //tem后处理文件
   ├── BSN_pem_preprocess.py       //pem前处理文件
   ├── BSN_tem_preprocess.py       //tem前处理文件
   ├── BSN_tem_pth2onnx.py     //tem pth 转换为onnx文件
   ├── BSN_pem_pth2onnx.py     //pem pth 转换为onnx文件
   ├── REAME.md 
   ```

3. 获取开源代码仓。

   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch   #获取开源代码仓
   git clone https://gitee.com/zheng-wengang1/onnx_tools     #获取优化工具
   ```



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型支持csv_mean_100图片的验证集。请用户需自行获取[csv_mean_100](https://github.com/wzmsltw/BSN-boundary-sensitive-network#download-datasets)数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。

   下载后将csv_mean_100放到开源仓代码目录，“./data/activitynet_feature_cuhk/”下

   ```
   ├── data
         ├── activitynet_feature_cuhk
               |--  csv_mean_100
   ```

   

2. 数据预处理。

   由于tem的后处理与pem的前处理有关，故pem的前处理置后执行。

   ```
   python3 BSN_tem_preprocess.py
   ```

   

3. 生成数据集info文件。

   生成bin文件的输入info文件。

   使用benchmark推理需要输入图片数据集的info文件，用于获取数据集。使用“gen_dataset_info.py”脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行“gen_dataset_info.py”脚本。

   ```
   python3 gen_dataset_info.py  tem ./output/BSN-TEM-preprocess/feature ./TEM-video-feature.info  400 100
   ```

   “tem”：生成的数据集文件格式。

   “./output/BSN-TEM-preprocess/feature”：预处理后的数据文件的**相对路径**。

   “./TEM-video-feature.info”：生成的数据集文件保存的路径。

   运行成功后，在当前目录中生成“TEM-video-feature.info”。




## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取权重文件：“model_final_f10217.pkl”。

   2. 导出onnx文件

      1. 代码转换为静态的onnx，需在代码中修改batchsize大小。

         a.执行命令编辑脚本。

         ```
         BSN_tem_pth2onnx.py
         #修改dummy_input = torch.randn(align_size,400,100) 中第一个参数为需要的batchsize
         执行:wq保存退出编辑。
         BSN_pem_pth2onnx.py
         #修改dummy_input = torch.randn(align_size,1000,32) 中第一个参数为需要的batchsize
         执行:wq保存退出编辑。
         
         ```
      
      2. 使用“pem_best.pth.tar”和“tem_best.pth.tar”导出onnx文件
      
         运行“BSN_tem_pth2onnx.py”脚本。
      
         ```
         python3 BSN_tem_pth2onnx.py --pth_path='${权重文件路径}' --onnx_path='${onnx文件的存放路径}'
         ```
      
      3. 优化onnx文件
      
         ```
         python3 TEM_onnx_conv1d2conv2d.py '${onnx文件的存放路径}' '${优化后的onnx文件存放路径}'
         ```
      
      4. 运行“BSN_pem_pth2onnx.py”脚本。
      
         ```
         python3 BSN_pem_pth2onnx.py --pth_path='${权重文件路径}' --onnx_path='${onnx文件的存放路径}'
         ```
      
      5. 使用ATC工具将ONNX模型转OM模型。
      
         1. 配置环境变量。
      
            ```
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```
      
            > **说明：** 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 (推理)](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fascend-computing%2Fcann-pid-251168373%3Fcategory%3Ddeveloper-documents%26subcategory%3Dauxiliary-development-tools)》。
      
         2. 执行命令查看芯片名称（${chip_name}）。
      
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
      
            
      
         3. 执行ACT命令
         
            
         
            ```
            atc --model=BSN_tem1_bs1.onnx --framework=5 --output=BSN_tem_bs1 --input_format=ND --input_shape="video:1,400,100"  --log=debug --soc_version=${chip_name}
            atc --model=BSN_pem_bs1.onnx --framework=5 --output=BSN_pem_bs1 --input_format=ND --input_shape="video_feature:1,1000,32"  --log=debug --soc_version=${chip_name}
            ```
         
            - 参数说明：
              
                 - --model：为ONNX模型文件。
                 - --framework：5代表ONNX模型。
                 - --output：输出的OM模型。
                 - --input_format：输入数据的格式。
                 - --input_shape：输入数据的shape。
                 - --log：日志级别。
              - --soc_version：处理器型号。
              
              运行成功后生成“BSN_tem_bs1.om”和“BSN_pem_bs1.om”模型文件。

2. 开始推理验证

   a. 安装ais_bench推理工具。

   执行命令增加工具可执行权限，并根据OS架构选择工具

   ```
   chmod u+x 
   ```

   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

   b. 执行推理。

   真实数据推理：
   
   ```
    python3.7 -m ais_bench --model BSN_tem_bs1.om --batchsize 1 --input="./output/BSN-TEM-preprocess/feature/"  --output ./ais_result --output_dirname result_tem_bs1 
   ```
   
   
   
   - 参数说明：
     - model：om文件路径。
     - input：输入数据。
     - batchsize：batchsize大小。
     - output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
     - --output_dirname：推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中
   
   推理后的输出默认在当前目录result下。

   
   c. 数据后处理。
   
   运行“BSN_tem_postprocess.py”脚本进行TEM模型后处理
   
   ```
    python3.7 BSN_tem_postprocess.py --TEM_out_path ./ais_result/result_tem_bs1/
   ```
   
   由于tem的后处理与pem的前处理有关，故pem的前处理置后执行。
   
   ```
    python3 BSN_pem_preprocess.py
   ```
   
   使用benchmark推理需要输入图片数据集的info文件，用于获取数据集。使用“gen_dataset_info.py”脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行“gen_dataset_info.py”脚本。
   
   ```
   python3 gen_dataset_info.py  pem ./output/BSN-PEM-preprocess/feature ./PEM-video-feature.info  1000 32
   ```
   
   “pem”：生成的数据集文件格式。
   
   “./output/BSN-PEM-preprocess/feature”：预处理后的数据文件的**相对路径**。
   
   “./PEM-video-feature.info”：生成的数据集文件保存的路径。
   
   运行成功后，在当前目录中生成“PEM-video-feature.info”。
   
   真实数据推理：
   
   ```
   python3.7 -m ais_bench --model BSN_pem_bs1.om --batchsize 1 --input="./output/BSN-PEM-preprocess/feature/"  --output ./ais_result --output_dirname result_pem_bs1 
   ```
   
   - 参数说明：
     - model：om文件路径。
     - input：输入数据。
     - batchsize：batchsize大小。
     - output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
     - --output_dirname：推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中
   
   运行“BSN_pem_postprocess.py”脚本进行PEM模型后处理
   
   ```
     python3.7 BSN_pem_postprocess.py --PEM_out_path ./ais_result/result_pem_bs1/
   ```
   
   d. 精度验证。
   
   将python2的代码，使用前转为python3转换脚本。
   
   ```
      2to3 -w ./BSN-boundary-sensitive-network.pytorch/Evaluation/eval_proposal.py
   ```
   
      ​	运行“BSN_eval.py”脚本，测试模型精度。
   
   ```
      python3 BSN_eval.py
   ```
   
      原模型精度74.34%






# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集       | 精度          | 性能     |
| -------- | ---------- | ------------ | ------------- | -------- |
| 310      | 1          | csv_mean_100 | AR100：74.34% | 6962.63  |
| 310      | 4          | csv_mean_100 | AR100：74.34% | 20458.91 |
| 310      | 8          | csv_mean_100 | AR100：74.34% | 30193.26 |
| 310      | 16         | csv_mean_100 | AR100：74.34% | 36202.23 |
| 310      | 32         | csv_mean_100 | AR100：74.34% | 35252.63 |
| 310      | 64         | csv_mean_100 | AR100：74.34% | 37288.98 |
| 310P     | 1          | csv_mean_100 | AR100：74.34% | 6197.32 |
| 310P     | 4          | csv_mean_100 | AR100：74.34% | 21052.35 |
| 310P     | 8          | csv_mean_100 | AR100：74.34% | 29687.91 |
| 310P     | 16         | csv_mean_100 | AR100：74.34% | 34617.79 |
| 310P     | 32         | csv_mean_100 | AR100：74.34% | 31336.91 |
| 310P     | 64         | csv_mean_100 | AR100：74.34% | 29685.42 |
| T4       | 1          | csv_mean_100 |               | 4962.25 |
| T4       | 4          | csv_mean_100 |               | 28301.35 |
| T4       | 8          | csv_mean_100 |               | 35080.33 |
| T4       | 16         | csv_mean_100 |               | 39867.73 |
| T4       | 32         | csv_mean_100 |               | 45102.31 |
| T4       | 64         | csv_mean_100 |               | 50327.56 |

**310P吞吐率计算公式（以bs1为例）**

TEM吞吐率：6049.12

PEM吞吐率：9152.86

BSN整体吞吐率为：1/(1/6049.12+1/9152.86)=3642.08