# RawNet2-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******


# 概述
RawNet2是用于说话人验证的模型，可提取原生音频的潜在特征，这是传统计算方法得不到的。
该模型提出了尺度向量用来自适应缩放特征图，同时引入了sinc卷积替代首层的普通卷积，实现了说话人验证的欺骗识别。

- 版本说明：
  ```
  url=https://github.com/asvspoof-challenge/2021
  commit_id=aae41bb3f4f38795ebf6901c7bbf3461e4ad662b
  model_name=RawNet2
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                     | 版本     | 环境准备指导                                                 |
| ------------------------------------------------------- |--------| ------------------------------------------------------------ |
| 固件与驱动                                                | 22.0.3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                    | 6.0.0  | -                                                            |
| Python                                                  | 3.7.5  | -                                                            |
| PyTorch                                                 | 1.10.1 | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \      | \                                                            |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/asvspoof-challenge/2021.git
   cd 2021/LA/Baseline-RawNet2/
   git reset --hard aae41bb3f4f38795ebf6901c7bbf3461e4ad662b
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```
   

3. 获取`OM`推理代码  
   将推理部署代码放在`2021/LA/Baseline-RawNet2`源码仓目录下。
   ```
   RawNet2_for_Pytorch
    ├── pth2onnx.py        放到2021/LA/Baseline-RawNet2下
    ├── modify_onnx.py     放到2021/LA/Baseline-RawNet2下
    ├── evaluation.py      放到2021/LA/Baseline-RawNet2下
    └── om_val.py          放到2021/LA/Baseline-RawNet2下
   ```   


## 准备数据集
- 该模型使用`LA`数据集进行精度评估，下载[LA数据集](https://datashare.ed.ac.uk/handle/10283/3336)，将下载的数据集放到`2021/LA/Baseline-RawNet2`源码仓目录下，文件结构如下：
   ```
   LA
   └── ASVspoof2019_LA_cm_protocols
      └── LA.cm.eval.trl.txt
   └── ASVspoof2019_LA_eval
      └── flac
        ├── LA_E_1000147.flac
        ├── LA_E_1000273.flac
        ├── ……
        └── LA_E_A9997819.flac
   ```


## 模型推理
### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   下载[权重下载](https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip)，放在`2021/LA/Baseline-RawNet2`目录下。
 

2. 导出`ONNX`模型  
   导`onnx`模型需要修改`2021/LA/Baseline-RawNet2/model.py`中的`SincConv`类，在该类的`forward`函数中增加一行，如下所示。  
   ```
   self.band_pass = torch.from_numpy(self.band_pass.numpy())   # 增加行，和下行缩进保持一致
   band_pass_filter=self.band_pass.to(self.device)  # 根据该行代码找到增加位置
   ```
   运行`pth2onnx.py`导出`ONNX`模型。  
   ```
   python3 pth2onnx.py --pth_model=pre_trained_DF_RawNet2.pth --onnx_model=rawnet2_bs1.onnx --batch_size=1
   ```
   修改导出的`onnx`模型，提升模型性能。
   请先安装 [onnx改图接口工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)   
   ```
   python3 -m onnxsim rawnet2_bs1.onnx rawnet2_bs1.onnx
   python3 modify_onnx.py --input_onnx=rawnet2_bs1.onnx --output_onnx=rawnet2_bs1.onnx
   ```

3. 使用`ATC`工具将`ONNX`模型转为`OM`模型  
   3.1 配置环境变量  
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   > **说明：**  
     该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   3.2 执行命令查看芯片名称（得到`atc`命令参数中`soc_version`）
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

   3.3 执行ATC命令  
   运行`atc.sh`导出`OM`模型，默认保存在`output`文件夹下。
   ```
   atc --model=rawnet2_bs1.onnx \
       --output=rawnet2_bs1 \
       --input_shape=input:1,64600 \
       --input_format=ND \
       --log=error \
       --framework=5 \
       --soc_version=Ascend310P3 \
       --optypelist_for_implmode="Sigmoid" \
       --op_select_implmode=high_performance
   ```
      - `atc`命令参数说明：
        -   `--model`：ONNX模型文件
        -   `--framework`：5代表ONNX模型
        -   `--output`：输出的OM模型
        -   `--input_format`：输入数据的格式
        -   `--input_shape`：输入数据的shape
        -   `--log`：日志级别
        -   `--soc_version`：处理器型号
        -   `--optypelist_for_implmode`：设置optype列表中算子的实现方式
        -   `--op_select_implmode`：选择算子是高精度实现还是高性能实现

    
### 2 开始推理验证

1. 安装`ais_bench`推理工具  
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

2. 执行推理 & 精度验证  
   运行`om_val.py`推理OM模型，得到模型精度结果。
   ```
   python3 om_val.py --om=rawnet2_bs1.om --batch=1
   ```

3. 性能验证  
   可使用`ais_bench`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能，参考命令如下：
   ```
   python3 -m ais_bench --model rawnet2_bs1.om --loop 1000 --batchsize 1 
   ```

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|   芯片型号   | Batch Size |   数据集    | 精度EER  | 精度min-tDCF |     性能     |
|:-----------:|:----------:|:--------:|:------:|:----------:|:----------:|
| Ascend310P3 |     1      |      LA  | 4.867% |   0.113    | 230.08 fps |
| Ascend310P3 |     4      |      LA  | 4.867% |   0.113    | 404.58 fps |
| Ascend310P3 |     8      |      LA  | 4.867% |   0.113    | 543.79 fps |
| Ascend310P3 |     16     |      LA  | 4.867% |   0.113    | 593.40 fps |
| Ascend310P3 |     32     |      LA  | 4.867% |   0.113    | 608.81 fps |
| Ascend310P3 |     64     |      LA  | 4.867% |   0.113    | 619.84 fps |
