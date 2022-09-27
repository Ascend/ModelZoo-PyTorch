# CTPN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)
- [推理环境准备](#ZH-CN_TOPIC_0000001126281699)
- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
  - [GPU上模型推理](#section741711594518)

- [模型推理性能和精度](#ZH-CN_TOPIC_0000001172201573)

  



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

CTPN是一种文字检测算法，它结合了CNN与LSTM深度网络，能有效的检测出复杂场景的横向分布的文字CTPN。作者开发了一种垂直锚定机制，可以联合预测每个固定宽度提议的位置和文本/非文本得分，大大提高了定位精度。序列提议通过循环神经网络自然连接，并与卷积网络无缝结合，形成一个端到端的可训练模型，这使得CTPN可以探索丰富的图像上下文信息，使其强大的检测极其模糊的文本。CTPN可以在多尺度和多语言文本上可靠地工作，而无需进一步的后处理，这与以前自下而上的方法需要多步后处理不同。CTPN只预测文本的竖直方向上的位置，水平方向的位置不预测，从而检测出长度不固定的文本。


- 参考实现：

  ```
  url = https://github.com/CrazySummerday/ctpn.pytorch.git
  branch=master 
  commit_id=99f6baf2780e550d7b4656ac7a7b90af9ade468f
  ```

- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```




## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x h x w | NCHW         |

    其中h,w分为10组：248x360, 280x550, 319x973, 458x440, 477x636, 631x471, 650x997, 753x1000, 997x744, 1000x462。


- 输出数据

  | 输出数据   | 大小                               | 数据类型 | 数据排布格式 |
  | ---------- | ---------------------------------- | -------- | ------------ |
  | class      | batchsize x (h//16) x (w//16) x 20 | FLOAT32  | ND           |
  | regression | batchsize x (h//16) x (w//16) x 20 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281699"></a>
- 该模型需要以下插件、驱动和依赖

  **表 1**  版本配套表	

  | 配套           | 版本     | 环境准备指导                                                 |
  |--------| -------- | ------------------------------------------------------------ |
  | 固件与驱动     | 1.0.15 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN           | 5.1.RC2 | -                                                            |
  | Python         | 3.7.5 | -                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 安装依赖
    ```
   pip3 install -r requirment.txt
   ```



## 获取源码

1. 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser/CTPN）。文件结构如下：

   ```
   ├── change_model.py           //onnx模型修改代码
   ├── config.py                 //确定的分档的分辨率大小以及每个档位的个数
   ├── ctpn_postprocess.py       //ctpn后处理文件
   ├── ctpn_preprocess.py        //ctpn前处理文件
   ├── ctpn_pth2onnx.py          //用于转换pth文件到onnx文件
   ├── image_kmeans.py           //确定相应的分档分辨率的聚类中心
   ├── performance_gpu.py        //计算gpu性能文件
   ├── README.md                 //readme文档
   ├── requirements.txt          //安装包信息
   ├── task_process.py           //任务处理文件，根据输入的不同模型完成相应的任务
   ```
   
2. 安装开源仓代码（在/home/HwHiAiUser/CTPN目录下）。
   ```
   git clone https://github.com/CrazySummerday/ctpn.pytorch.git -b master
   cd ctpn.pytorch
   git reset 99f6baf2780e550d7b4656ac7a7b90af9ade468f –hard
   cd ..
   ```

   

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型使用ICDAR2013数据，获取[数据集](https://rrc.cvc.uab.es/?ch=2)及相应[评测方法代码](https://rrc.cvc.uab.es/standalones/script_test_ch2_t1_e2-1577983067.zip)。在本目录（如/home/HwHiAiUser/CTPN）**新建data和script文件夹**。将数据集解压为Challenge2_Test_Task12_Images文件夹，并放入data文件夹下。将测评方法代码解压放入script文件夹。目录结构如下：

   ```
   data
   ├──Challenge2_Test_Task12_Images
   script
   ├──gt.zip
   ├──readme.txt
   ├──rrc_evaluation_funcs_1_1.py
   ├──script.py
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。因为该模型根据图片输入形状采用分档输入，一共分为了10档，因此需要生成不同分辨率的预处理文件，为简化步骤、避免浪费不必要的时间，直接将相应的预处理程序放在任务处理的“task_process.py”脚本中，该脚本会自动删除和创建数据预处理的文件夹，以及调用预处理“ctpn_preprocess.py”程序。

   ```
   python3 task_process.py --mode='preprocess' --src_dir='./data/Challenge2_Test_Task12_Images'
   
   --mode：执行的模块。
   --src_dir：数据集路径。
   ```

   运行上述命令后会在data目录下生成10个目录：images_bin_248x360, images_bin_280x550, images_bin_319x973, images_bin_458x440, images_bin_477x636, images_bin_631x471, images_bin_650x997, images_bin_753x1000, images_bin_997x744, images_bin_1000x462。每个目录下含有相同形状的图片对应的数据文件，例如images_bin_248x360目录下存放形状为248x360的经过预处理的图片数据。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       权重文件在./ctpn.pytorch/weights/ 目录下，文件名称为ctpn.pth。

   2. 导出onnx文件。

      1. 使用ctpn_pth2onnx.py导出onnx文件。

         运行ctpn_pth2onnx.py脚本。

         ```
         python3 ctpn_pth2onnx.py --pth_path='./ctpn.pytorch/weights/ctpn.pth' --onnx_path='ctpn.onnx'
      
         --pth_path：pth权重路径。
         --onnx_path：onnx路径。
         ```

         执行成功后获得ctpn_248x360.onnx, ctpn_280x550.onnx, ctpn_319x973.onnx, ctpn_458x440.onnx, ctpn_477x636.onnx, ctpn_631x471.onnx, ctpn_650x997.onnx, ctpn_753x1000.onnx, ctpn_997x744.onnx, ctpn_1000x462.onnx文件。

      2. 使用task_process.py优化ONNX文件。

         ```
         python3 task_process.py --mode='change model'
         
         --mode：执行的模块。
         ```
         
         执行成功后获得ctpn_change_248x360.onnx, ctpn_change_280x550.onnx, ctpn_change_319x973.onnx, ctpn_change_458x440.onnx, ctpn_change_477x636.onnx, ctpn_change_631x471.onnx, ctpn_change_650x997.onnx, ctpn_change_753x1000.onnx, ctpn_change_997x744.onnx, ctpn_change_1000x462.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。
         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称。

         ```
         npu-smi info
         回显如下：
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 16.6         57                0    / 0              |
         | 0       0         | 0000:3B:00.0    | 0            928  / 21534                            |
         +===================+=================+======================================================+
         ```
         
      3. 执行ATC命令。

         ```
         atc --framework=5 --model=ctpn_change_1000x462.onnx --output=ctpn_bs1 --input_format=NCHW --input_shape="image:1,3,-1,-1" --dynamic_image_size="248,360;280,550;319,973;458,440;477,636;631,471;650,997;753,1000;997,744;1000,462" --log=error --soc_version=Ascend${chip_name}
         ```
         - 参数说明：
           - --framework：5代表ONNX模型。
           - --model：为ONNX模型文件。
           - --output：输出模型文件名称。
           - --input_format：输入数据格式。
           - --input_shape：模型输入数据的shape。
           - --dynamic_image_size：设置输入图片的动态分辨率参数。适用于执行推理时，每次处理图片宽和高不固定的场景。
           - --log：设置ATC模型转换过程中显示日志的级别。
           - --soc_version：模型转换时指定芯片版本。
      
         运行成功后生成对应芯片版本的.om模型文件。

2. 开始推理验证。

   2. 安装推理工具ais_infer。

      在当前目录下（如/home/HwHiAiUser/CTPN）执行以下命令

      ```
      git clone https://gitee.com/ascend/tools.git
      cd tools/ais-bench_workload/tool/ais_infer/backend/
      pip wheel ./
      pip install ./aclruntime-0.0.1-cp37-cp37m-linux_x86_64.whl
      ```

      然后返回到/home/HwHiAiUser/CTPN目录。

   3. 创建结果输出目录。

      在当前目录下（如/home/HwHiAiUser/CTPN）执行以下命令

      ```
      mkdir result
      cd result
      mkdir inf_output
      mkdir dumpOutput_device0
      cd ..
      ```

      输出目录的结构如下：

      ```
      result
      ├──inf_output
      ├──dumpOutput_device0
      ```

      ais_infer的推理结果会输出到result/inf_output/目录下，由于模型是按照输入图片形状进行分档处理的，result/inf_output/下会有多个输出目录，这些目录都是以日期命名，为了方便处理，后面需要将这些目录下的模型推理结果文件都移动到result/dumpOutput_device0/目录下。

   3. 执行推理。

      ```
      python3 task_process.py  --mode='ais_infer' --machine='Ascend310P' --interpreter='python'
      
      --mode：执行的模块。
      --machine：芯片名称。'Ascend310P' 或 'Ascend310'
      --interpreter: 执行推理命令的解释器
      ```

      在推理之前，删除./result/inf_output/和./result/dumpOutput_device0/里的文件和文件夹，防止受到上次推理的影响。task_process.py会将分散在./result/inf_output/里的模型输出文件移动到./result/dumpOutput_device0/目录下。命令执行成功后会在./result/dumpOutput_device0/目录下获得模型的输出文件，并且在屏幕上输出性能数据。

      性能计算方式：

      设输入数据根据宽高的不同分为 $n$ 组，第 $i$ 组的性能为 $f_i$，第 $i$ 组的数据集大小为 $s_i$，则模型的综合性能的计算公式为：
      $$
      performance = \frac{\sum_i^n f_i*s_i}{\sum_i^ns_i}
      $$

   5. 精度验证。

      1. 创建输出目录。

         ```
         cd data
         rm -rf predict_txt
         mkdir predict_txt
         cd ..
         ```

      2. 执行后处理脚本“ctpn_postprocess.py”计算精度并形成数据压缩包。

         ```
         python3 ctpn_postprocess.py --imgs_dir=data/Challenge2_Test_Task12_Images --bin_dir=result/dumpOutput_device0 --predict_txt=data/predict_txt
         
         --imgs_dir：数据集路径。
         --bin_dir：精度数据路径。
         --predict_txt：后处理输出文件路径。
         
         rm -rf script/predict_txt.zip
         cd data/predict_txt
         zip -rq predict_txt.zip ./*
         mv predict_txt.zip ../../script/
         cd ../..
         ```

         执行上述命令后，会在./script目录下生成一个数据压缩包predict_txt.zip。

      3. 计算精度数据。

         ```
         python3 script/script.py -g=script/gt.zip –s=script/predict_txt.zip
         
         -g：label的路径。
         -s：模型预测数据压缩包路径。
         ```

         执行成功后会在屏幕上输出模型精度数据。

## GPU上模型推理<a name="section741711594518"></a>

1. 执行上述环境准备、获取源码和获取数据集（不用进行数据预处理）步骤。

2. 创建输出目录。

   ```
   cd data
   rm -rf predict_txt
   mkdir predict_txt
   cd ..
   ```

3. 执行推理。

   ```
   python3 ctpn_postprocess.py --model=pth --imgs_dir=data/Challenge2_Test_Task12_Images  --pth_txt=data/pth_txt
   
   --model：模型文件类型。
   --imgs_dir：数据集路径。
   --pth_tx：输出文件路径
   ```

   执行成功后会在输出文件路径上生成模型输出文件，在屏幕上会输出性能数据。

4. 精度验证。

   1. 形成数据压缩包

      ```
      rm -rf script/pth_txt.zip
      cd data/pth_txt
      zip -rq pth_txt.zip ./*
      mv pth_txt.zip ../../script/
      cd ../..
      ```

      执行成功后会将模型输出文件压缩并移动到script目录下。

   2. 计算精度数据。

      ```
      python3 script/script.py -g=script/gt.zip –s=script/pth_txt.zip
      
      -g：label的路径。
      -s：模型预测数据压缩包路径。
      ```

      执行成功后会在屏幕上输出模型精度数据。

   

# 模型推理性能和精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

| 芯片型号   | Batch Size | 数据集    | 精度                                              | 性能            |
| ---------- | ---------- | --------- | ------------------------------------------------- |---------------|
| Ascend310P | 1          | ICDAR2013 | precision: 86.84%   recall: 75.05%  hmean: 80.51% | 148.3098 fps  |
| Ascend310  | 1          | ICDAR2013 | precision: 86.84%   recall: 75.05%  hmean: 80.51% | 93.4589   fps |
| T4         | 1          | ICDAR2013 | precision: 87.41%   recall: 75.60%  hmean: 81.08% | 73.6914   fps |

