## 取源码及安装

**步骤 1** 单击“立即下载”，下载源码包。

**步骤 2** 下载Pelee开源代码
```shell
git clone https://github.com/yxlijun/Pelee.Pytorch
git checkout -b 1eab4106330f275ab3c5dfb910ddd79a5bac95ef
```


**步骤 3** 上传Pelee.patch到和源码同一级目录下，执行patch命令。

```shell
patch -p0 < Pelee.patch
```

**步骤 4** 在Pelee.Pytorch目录下编译开源代码

```shell
bash make.sh
```

**步骤 5** 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。

├── acl_net.py
├── ReadMe.md 
├── atc.sh                //onnx模型转换om模型脚本 
├── Pelee.patch             //Pelee模型修改patch文件 
└── pth2onnx.py             //用于转换pth模型文件到onnx模型文件 

**步骤 6** 拷贝源码和文件到Pelee.Pytorch目录下。

## 准备数据集

**步骤 1** 获取原始数据集。

本模型该模型使用VOC2007的4952张验证集进行测试，请用户自行获取该数据集，上传并解压数据集到服务器任意目录。

**步骤 2** 数据预处理。

数据预处理复用Pelee开源代码，具体参考test.py代码。

## 模型推理

**步骤 1** 模型转换。

本模型基于开源框架PyTorch训练的Pelee进行模型转换。

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

- [x] 获取权重文件。

  在PyTorch开源预训练模型中获取Pelee_VOC.pth权重文件。

- [x] 导出onnx文件。

  pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令。

  ```shell
  python3.7.5 pth2onnx.py --config ./configs/Pelee_VOC.py -m ./Pelee_VOC.pth -o ./pelee_dynamic_bs.onnx
  ```

  第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。

  运行成功后，在当前目录生成pelee_dynamic_bs.onnx模型文件。

 

​	**说明:** 使用ATC工具将.onnx文件转换为.om文件，需要.onnx算子版本需为13。在pth2onnx.py脚本中torch.onnx.export方法中的输入参数opset_version的值需为13。

- [x] 用onnx-simplifier简化模型

  ```shell
  python3.7.5 -m onnxsim pelee_dynamic_bs.onnx pelee_dynamic_bs_sim.onnx --input-shape 1,3,304,304
  ```

- [x] 改图优化

​	修改softmax节点，在softmax前插入transpose

  ```shell
  python3.7.5 softmax.py pelee_dynamic_bs_sim.onnx pelee_dynamic_bs_modify.onnx
  ```

![img](file:///C:\Users\C00444~1\AppData\Local\Temp\ksohtml124560\wps2.jpg) 

- softmax.py修改模型节点需要和onnx模型中Softmax节点name保持一致。如果执行脚本报错时参考onnx图中Softmax节点的name。

- ONNX改图依赖om_gener工具，下载链接：https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Official/nlp/VilBert_for_Pytorch/om_gener

 


- [x] 使用ATC工具将ONNX模型转OM模型。

  a. 修改atc.sh脚本，通过ATC工具使用脚本完成转换，具体的脚本示例如下：

  ${chip_name}可通过`npu-smi info`指令查看
     
   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
  
  ```shell
  # 配置环境变量 
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  
  # 使用二进制输入时，执行如下命令。不开启aipp，用于精度测试
  ${install_path}/atc/bin/atc --model=./pelee_dynamic_bs_modify.onnx --framework=5 --output=pelee_bs1 --input_format=NCHW --input_shape="image:1,3,304,304" --log=info --soc_version=Ascend${chip_name} --enable_small_channel=1 # Ascend310P3
  
  # 使用二进制输入时，执行如下命令。开启aipp，用于性能测试
  ${install_path}/atc/bin/atc --model=./pelee_dynamic_bs_modify.onnx --framework=5 --output=pelee_bs32 --input_format=NCHW --input_shape="image:32,3,304,304" --log=info --soc_version=Ascend${chip_name} --insert_op_conf=aipp.config --enable_small_channel=1 # Ascend310P3
  ```



**说明**

该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

n 参数说明：

□ --model：为ONNX模型文件。

□ --framework：5代表ONNX模型。

□ --output：输出的OM模型。

□ --input_format：输入数据的格式。

□ --input_shape：输入数据的shape。

□ --log：日志等级。

□ --soc_version：部署芯片类型。

□ --insert_op_conf=aipp.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。



​	b. 执行atc.sh脚本，将.onnx文件转为离线推理模型文件.om文件。

```shell
bash atc.sh
```

运行成功后生成pelee**_**bs1.om用于二进制输入推理的模型文件，生成的pelee**_**bs32.om用于推理性能测试。

- [x]  开始推理验证。

  设置pyACL环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

 

## 执行推理

```shell
python3.7.5 test.py --dataset VOC  --config ./configs/Pelee_VOC.py --device_id 0 --model pelee_bs1.om
```

