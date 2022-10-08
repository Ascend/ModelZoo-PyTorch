# EfficientNet-b7模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 安装开源模型代码仓 

```
pip3.7 install efficientnet-pytorch==0.7.1
```

## 2 准备数据集

1. 获取原始数据集
   
   本模型使用ImageNet 50000张图片的验证集，请参考[Pytorch官方文档](https://github.com/pytorch/examples/tree/main/imagenet)下载和处理数据集
   
   处理完成后获得分目录的图片验证集文件，目录结构类似如下格式：
   ```text
    imagenet/val/
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ......
    ├── ......
    ```

2. 数据预处理

    将原始数据集转换为模型输入的二进制数据，请预留充足的磁盘空间（约205G）
    ```
    python3.7 preprocess.py --dataset_path=imagenet/val --save_path=data_bin --batch_size=32 --image_size=600
    ```
    按照指定的`--batch_size`，每32个图像对应生成一个二进制bin文件。运行成功后，在当前目录下生成`data_bin`二进制文件夹以及`label.txt`文件。

## 3 模型生成

1. 生成onnx模型

   1. 获取权重文件

        导出onnx时自动下载
    
   2. 导出onnx文件

        ```
        python3.7 pth2onnx.py --version=7
        ```
   3. 优化onnx文件

        ```
        python3.7 -m onnxsim efficientnet_b7_dym_600.onnx efficientnet_b7_dym_600_sim.onnx --dynamic-input-shape --input-shape 1,3,600,600
        ```

2. 将onnx转为om模型

   1. 配置环境变量
        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

   2. 使用ATC工具将onnx模型转om模型

        `${chip_name}`可通过`npu-smi info`指令查看

        ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

        ```
        bash atc.sh efficientnet_b7_dym_600_sim.onnx efficientnet_b7_32_600_sim Ascend${chip_name} # Asecend310P3
        ```
        运行成功后生成efficientnet_b7_32_600_sim.om模型文件

## 4 离线推理 

1. 安装ais_infer工具

    参考链接安装：https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer
    
    可选操作：
    
    * 安装完成后，修改安装目录下的`ais_infer.py`文件，在首行添加`#!/usr/bin/env python3.7`;
    * 为`ais_infer.py`添加可执行权限：`chmod +x ais_infer.py`;
    * 将`ais_infer.py`所在目录添加到`PATH`环境变量中：`PATH=<ais_infer.py所在路径>:$PATH`.
    
    设置完成之后可以直接通过`ais_infer.py`命令调用工具。若不执行上述可选操作，则需要通过`python3.7 ais_infer.py ...`方式调用工具。
   
2. 模型推理

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   mkdir outputs
   python3.7 ais_infer.py --model efficientnet_b7_32_600_sim.om --batchsize 32 --input data_bin/ --output outputs/ --outfmt=BIN --device 0
   ```
   运行成功后会在`outputs/{dir}`下生成推理输出的bin文件和推理性能测试结果`summary.json`。

3. 精度验证

   首先将`outputs/{dir}`目录下的`summary.json`文件移动到其他目录，然后将`outputs/{dir}`改为实际目录，统计推理输出的Top 1 Accuracy
   ```
   python3.7 postprocess.py --output_dir=outputs/{dir} --label_path=label.txt
   ```
   

**评测结果：**   

|    模型    |    官网pth精度    | 310P离线推理精度 | gpu性能 | 310P性能  |
| :--------: | :---------------: |:----------:| :-----: |:-------:|
| EfficientNet-b7 bs32 | acc: 84.4 | acc: 84.46 | 55.4fps  | 87.3fps |

