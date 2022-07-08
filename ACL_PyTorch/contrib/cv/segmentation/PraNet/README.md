# PraNet模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取，安装开源模型代码。

```shell
git clone https://github.com/DengPingFan/PraNet.git -b master
cd PraNet
git reset --hard f697d5f566a4479f2728ab138401b7476f2f65b9
patch -p1 < ../PraNet_perf.diff
<!--
因开源代码仓使用matlab评测，故需从(https://github.com/plemeri/UACANet)获取pytorch实现的评测脚本eval_functions.py，并将其放在utils目录下
将./lib/PraNet_Res2Net.py的res2net50_v1b_26w_4s(pretrained=True)修改为res2net50_v1b_26w_4s(pretrained=False)
-->
cd ..
```
3.获取权重文件  
[PraNet训练的pth权重文件](https://drive.google.com/file/d/1pUE99SUQHTLxS9rabLGe_XTDwfS6wXEw/view)

4.数据集     
[kvasir](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)获取TestDataset.zip并解压出Kvasir，将其放到/root/datasets/目录下，即/root/datasets/Kvasir

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310p上执行时使用npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/Kvasir
```
- 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   导出onnx文件。

   使用pth文件导出onnx文件，运行PraNet_pth2onnx.py脚本。

   ```
   python3.7 PraNet_pth2onnx.py   ./PraNet-19.pth  ./PraNet-19.onnx batch_size
   ```

   参数说明：

   - “./PraNet-19.pth”：输入文件目录。
   - “./PraNet-19.onnx”：输出文件目录。
   - "batch_size":指定输出文件batch_size

   获得“PraNet-19.onnx”文件。

   使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。

   使用ATC工具将ONNX模型转OM模型。

   1. 配置环境变量

      ```
      set env.sh 
      ```
      
   2. 执行ATC命令。
   
      ${chip_name}可通过`npu-smi info`指令查看，例：310P3
   
      ```
      atc --framework=5 --model=PraNet-19bs1.onnx --output=PraNet-19_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,352,352"  --log=debug  --soc_version=Ascend${chip_name}
      ```
   
      该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      - 参数说明：
   
     - --model：为ONNX模型文件。
       - --framework：5代表ONNX模型。
       - --output：输出的OM模型。
       - --input_format：输入数据的格式。
       - --input_shape：输入数据的shape。
       - --log：日志级别。
       - --soc_version：处理器型号。
   
   
   
- 开始推理验证。

   

   1. 使用Benchmark工具进行推理。

      执行以下命令增加Benchmark工具可执行权限，并根据OS架构选择工具，如果是X86架构，工具选择benchmark.x86_64，如果是Arm，选择benchmark.aarch64 。

      ```
      chmod u+x benchmark.${arch}
      ```
      
      - 二进制输入
      
        ```
         ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./PraNet-19_bs1.om -input_text_path=./pre_bin.info -input_width=352 -input_height=352 -output_binary=True -useDvpp=False
        ```
      
        - 参数说明：
      
          - -model_type：模型类型。
          - -om_path：om文件路径。
          - -device_id：NPU设备编号。
          - -batch_size：参数规模。
          - -input_text_path：图片二进制信息。
          - -input_width：输入图片宽度。
          - -input_height：输入图片高度。
          - -useDvpp：是否使用Dvpp。
          - -output_binary：输出二进制形式。
      
          推理后的输出默认在当前目录“result”下。
      
          
      
          执行./benchmark*.x86_64*工具请选择与运行环境架构相同的命令。参数详情请参见《[CANN 推理benchmark工具用户指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
   2. 精度验证。

      调用“PraNet_postprocess.py”和“Eval.py”脚本，可以获得Accuracy数据。

      ```
      python3.7 PraNet_postprocess.py  /root/datasets/Kvasir ./result/dumpOutput_device0 ./bs1_test/Kvasir/
      python3.7 Eval.py   /root/datasets/Kvasir ./bs1_test/Kvasir/  ./bs1_test/result_bs1.json 
      ```

      “/root/datasets/Kvasir”：数据集路径。

      “./result/dumpOutput_device0”：推理结果路径。

      “./bs1_test/Kvasir/”：输出图片路径。

      “./bs1_test/result_bs1.json” :输出精度数据路径。

 **评测结果：**   

| ThroughOutput |   310   |  310P   |   T4    | 310P/310 | 310P/T4 |
| :-----------: | :-----: | :-----: | :-----: | :------: | ------- |
|      bs1      | 197.998 | 257.267 | 264.634 |  1.304   | 0.973   |
|      bs4      | 245.410 | 320.093 | 381.105 |  1.305   | 0.839   |
|      bs8      | 240.005 | 293.541 | 403.570 |  1.289   | 0.727   |
|     bs16      | 199.660 | 249.195 | 428.659 |  1.247   | 0.581   |
|     bs32      | 158.888 | 191.656 | 427.253 |  1.206   | 0.448   |

|  模型精度  |          310          |         310P          |
| :--------: | :-------------------: | :-------------------: |
| PraNet bs1 | mDec:0.894;mIoU:0.836 | mDec:0.894;mIoU:0.836 |
| PraNet bs4 | mDec:0.894;mIoU:0.836 | mDec:0.894;mIoU:0.836 |

