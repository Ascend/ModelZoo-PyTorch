# Res2Net_v1b_101模型测试指导

- [Res2Net_v1b_101模型测试指导](#res2net_v1b_101模型测试指导)
  - [1 文件说明](#1-文件说明)
  - [2 设置环境变量](#2-设置环境变量)
  - [3 端到端推理步骤](#3-端到端推理步骤)
    - [3.1 下载代码](#31-下载代码)
    - [3.2 om模型转换](#32-om模型转换)
    - [3.3 om模型推理](#33-om模型推理)

------

## 1 文件说明
```
Res2Net_v1b_101_for_PyTorch
    ├── get_info.py                      // 生成推理输入的数据集二进制info文件或jpg info文件
    ├── pth2onnx.py                      // 用于转换pth模型文件到onnx模型文件
    ├── diff.patch                       // 修改开源代码的patch文件
    ├── imagenet_torch_preprocess.py     // imagenet数据集预处理，生成图片二进制文件
    ├── README.md
    ├── atc.sh                           // onnx模型转换om模型脚本
    └── vision_metric_ImageNet.py        // 验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy

```

## 2 设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 3 端到端推理步骤

### 3.1 下载代码  
git clone 开源仓 https://github.com/Res2Net/Res2Net-PretrainedModels ，切换到所需tag。
```shell
git clone https://github.com/Res2Net/Res2Net-PretrainedModels.git
cd Res2Net-PretrainedModels
git reset 1d51000f3340fb61b4 --hard
git apply diff.patch
```

### 3.2 om模型转换

通过pth2onnx.py脚本转化为onnx模型

```shell
# 直接导出原始ONNX
python3.7 pth2onnx.py -m ./res2net101_v1b_26w_4s-0812c246.pth -o ./res2net.onnx

# 导出NPU上优化后的ONNX
python3.7 pth2onnx.py -m ./res2net101_v1b_26w_4s-0812c246.pth -o ./res2net.onnx --optimizer
```


利用ATC工具转换为om模型， ${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
   
  ```shell
  bash atc.sh Ascend${chip_name} # Ascend310P3
  ```

### 3.3 om模型推理

（1） 数据集预处理

  数据预处理，把ImageNet 50000张图片转为二进制文件（.bin）

   ```shell
   python3.7 imagenet_torch_preprocess.py res2net101 /home/HwHiAiUser/dataset/ImageNet/ILSVRC2012_img_val ./prep_bin
   ```
  生成数据集info文件

   ```shell
   python3.7 get_info.py bin ./prep_bin ./BinaryImageNet.info 224 224
   ```
（2）推理
  配置环境变量，运行benchmark工具进行推理，参数说明参见 [cann-benchmark](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 如果前面配置过，这里不用执行
  ./benchmark -model_type=vision -om_path=resnet_bs16.om -device_id=0 -batch_size=16 -input_text_path=BinaryImageNet.info -input_width=256 -input_height=256 -useDvpp=false -output_binary=false
  ```

（3）统计Accuracy值
  精度验证，调用vision_metric_ImageNet.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中

   ```shell
   python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
   ```
