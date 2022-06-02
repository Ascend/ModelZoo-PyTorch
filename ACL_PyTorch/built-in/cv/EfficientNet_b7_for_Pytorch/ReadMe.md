# EfficientNet-b7模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 安装开源模型代码仓 

```
pip3.7 install efficientnet_pytorch==0.7.1
```

## 2 准备数据集

1. 获取原始数据集
   
   本模型使用ImageNet 50000张图片的验证集，请参考Pytorch官方文档下载和处理数据集：https://github.com/pytorch/examples/tree/main/imagenet

2. 数据预处理

    将原始数据集转换为模型输入的数据
    ```
    python3.7 preprocess.py --dataset_path=imagenet/val --save_path=data_bin --batch_size=32 --image_size=600
    ```
    每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成data_bin二进制文件夹以及label.txt

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

   1. 安装mindx-toolbox工具
        
        https://support.huawei.com/enterprise/zh/ascend-computing/mindx-pid-252501207/software

   2. 配置环境变量
        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

   3. 使用ATC工具将onnx模型转om模型

        ```
        bash atc.sh
        ```
        运行成功后生成efficientnet_b7_32_600_sim.om模型文件

## 4 离线推理 

1. 安装msame工具

    参考链接安装：https://gitee.com/ascend/tools/tree/master/msame
   
2. 模型推理

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ./msame --model=efficientnet_b7_32_600_sim.om --input=data_bin --device=0 --output=outputs --outfmt=BIN
   ```
   运行成功后会在outputs/{dir}下生成推理输出的bin文件

3. 精度验证

     将outputs/{dir}改为实际目录
   ```
   python3.7 postprocess.py --output_dir=outputs/{dir} --label_path=label.txt
   ```

**评测结果：**   

|    模型    |    官网pth精度    | 310P离线推理精度 | gpu性能 |         310P性能         |
| :--------: | :---------------: | :-----------------: | :-----: | :---------------------: |
| EfficientNet-b7 bs32 | acc: 84.4 |  acc: 84.14  | 55.4fps  | 89.7fps |

