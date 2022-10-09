# StarGAN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

 

  



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

StarGAN是 Yunjey Choi 等人于 17年11月 提出的一个模型。该模型可以实现 图像的多域间的迁移（作者在论文中具体应用于人脸属性的转换）。在 starGAN 之前，也有很多 GAN模型 可以用于 image-to-image，比如 pix2pix（训练需要成对的图像输入），UNIT（本质上是coGAN），cycleGAN（单域迁移）和 DiscoGAN。而 starGAN 使用 一个模型 实现 多个域 的迁移，这在其他模型中是没有的，这提高了图像域迁移的可拓展性和鲁棒性。




- 参考实现：

  ```
  url=https://github.com/yunjey/stargan
  commit_id=94dd002e93a2863d9b987a937b85925b80f7a19f
  model_name=StarGAN
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

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | img    | RGB_FP32 | batchsize x  3 x 128 x 128| NCHW         |
   | Attr    | INT | batchsize x 5| NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 3 x 128 x 128 | RGB_FP32  | NCHW           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

**表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |

  **表 2**  环境依赖表
| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| ONNX                                                         | 1.8.0 | -                                                            |
| Numpy                                                         | 1.21.1 | -                                                            |
| TorchVision                                                         | 0.6.0 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码
   源码目录结构：
   ```
   ├── unzip_dataset.sh             //解压数据集
   ├── eval_bs1_perf.sh       //310Pbs1验收脚本
   ├── eval_bs16_perf.sh      //310Pbs16验收脚本
   ├── pth2om.sh                //310P生成om文件
   ├── StarGAN_pre_processing.py  //预处理
   ├── StarGAN_pth2onnx.py          //用于转换pth文件到onnx文件
   ├── StarGAN_post_processing.py //用于转换txt文件到jpg文件
   ├── model.py                        //定义模型的文件
   ├── solver.py                      //定义模型的文件
   ├── data_loader.py                //定义模型的文件
   ├── ReadMe.md 
   ```

2. 安装依赖。

   ```
   source ${CANN_INSTALL_PATH}/set_env.sh
   pip install --force-reinstall  aclruntime-0.0.1-cp37-cp37m-linux_x86_64.whl
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   数据集默认路径为 `./data/celeba.zip` ，使用脚本 `unzip_dataset.sh` 解压数据集。

   ```
   bash unzip_dataset.sh
   ```
   数据目录结构请参考：
   ```
   ├──celeba
    ├──images
    ├──list_attr_celeba.txt
   ```
2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。
   使用脚本 `StarGAN_pre_processing.py` 获得二进制 bin 文件和基准的图片结果。

  

   ```
   python3.7 StarGAN_pre_processing.py --mode test  --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                  --model_save_dir './models' --result_dir './result_baseline' \
                  --attr_path './data/celeba/list_attr_celeba.txt' --celeba_image_dir './data/celeba/images'  --batch_size 16
   
   ```
   > 本处以--batch_size 16为例，其他batchsize再此修改


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       pth 权重文件默认路径为  ./models/200000-G.pth

   2. 导出onnx文件。

      1. 使用pth2om.sh导出onnx文件。

         运行pth2om.sh脚本。

            ```
            bash pth2om.sh './models/200000-G.pth'
            ```

         获得StarGAN.onnx文件。

   

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

            ```
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```
        上一步已经生成OM模型
         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称($\{chip\_name\})，确保device空闲。

         ```
         npu-smi info
         ```
         ```
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
         ```
          atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs1 --input_format=NCHW \
            --input_shape="real_img:1,3,128,128;attr:1,5" --log=debug --soc_version=$\{chip\_name\}      
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

        运行成功后生成StarGAN_bs1.om模型文件。



2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。
      以batchsize=16 为例子
      ```
      python3  ais_infer.py --model ./StarGAN_bs16.om  --input ./bin/img,./bin/attr --output ./op  --outfmt TXT  --batchsize 16
      ```

      -   参数说明：

           -   StarGAN：模型类型。
           -   --model ./StarGAN_bs64.om：om文件路径。
           
		

      推理后的输出默认在当前目录result下。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。
   
      调用 ` StarGAN_post_processing.py` 来进行后处理，把输出的 txt 文件转换为输出图像。
      ```
      python3.7 StarGAN_post_processing.py --folder_path './op/2022_09_05-07_37_38' --batch_size 16
      ```

     详细的结果输出在 `./output_[yourBatchSize]/jpg` 文件夹中，可以和 `result_baseline` 文件夹下的在线推理结果做对比。可以发现各个 batchsize 的离线推理生成的图片与基准基本一致。


   d.  性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 ais_infer.py --model ./StarGAN_bs1.om  --output ./ --outfmt BIN --loop 5 --batchsize 1
      python3 ais_infer.py --model ./StarGAN_bs64.om  --output ./ --outfmt BIN --loop 5 --batchsize 64
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。


| Batch Size | 310 (FPS/Card) | 310P (FPS/Card)| T4 (FPS/Card) | 310/T4   | 310P/310  |  310P/T4  | 
| ---------- | -------------- | ------------- | ------------- | -------- | -------- | -------- |
| 1          | *189.753*      | *757.576*     | *187.135*     | *101.4%* | *396.8%* | *375.0%* |
| 4          | *201.207*      | *923.788*     | *203.666*     | *98.80%* | *458.3%* | *451.9%* |
| 8          | *199.913*      | *984.010*     | *219.700*     | *91.00%* | *491.3%* | *457.2%* |
| 16         | *200.986*      | *1015.911*     | *235.980*     | *85.17%* | *506.2%* | *448.3%* |
| 32         | *200.986*      | *991.633*     | *202.280*     | *99.36%* | *493.3%* | *490.2%* |
| 64         | *201.307*      | *1040.31*     | *195.670*     | *102.8%* | *516.7%* | *531.6%* |

精度测试：

baseline
![输入图片说明](https://foruda.gitee.com/images/1661163438787898836/屏幕截图.png "屏幕截图.png")

310P：

bs1:

![输入图片说明](https://foruda.gitee.com/images/1662535055936672123/ab08df50_5666861.png "屏幕截图")

bs64:

![输入图片说明](https://foruda.gitee.com/images/1662535255442916612/3a7bc395_5666861.png "屏幕截图")
