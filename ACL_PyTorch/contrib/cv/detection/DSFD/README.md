



# 概述

FaceDetection-DSFD是通用场景下的人脸检测模型，采用FSSD+Resnet的网络结构，加入FEM 模块进一步增强不同感受野的目标特征，实现高准确率的人脸检测

## 1.环境准备



1.安装必要的依赖

```python
pip install -r requirements.txt
source env.sh
```

2.执行 eval_tols/dsfd_acc_eval.py 之前先执行以下命令

```
cd eval_tools
python setup.py build_ext --inplace
```

3.获取权重文件

[pth模型链接](链接：https://pan.baidu.com/s/1DKNAKusuSh8O_91xvpCtWw  提取码：i468)  下载后放在根目录下

4.获取推理图像集 放在 opt/npu/目录下

[推理图像数据集](链接：https://pan.baidu.com/s/1KvpfjR0U8KUJnY7Gw5vLnQ  提取码：e3lu)

5.获取benchmark工具

将benchmark.x86_64或benchmark.aarch64放到主目录下

```
chmod a+x benchmark.x86_64
```

6.进行数据预处理

```python
python dsfd_preprocess.py --src_path '/opt/npu/WIDERFace/WIDER_val/images/' #主目录下产生info_result.info文件
```



## 2.模型转换

1.进行pth转onnx模型

```
cd test
python dsfd_pth2onnx.py --model_path '../dsfd.pth'
```

[onnx文件链接](链接：https://pan.baidu.com/s/1HR5Ur5-KjNYlVJnJ6JOdVg  提取码：yqep) 生成的onnx模型文件在test文件夹下

2.进行onnx模型转om模型

cd到test目录下执行以下命令

```
bash onnx2om.sh
```

生成的om模型在上一层 onnx2om 文件夹下

## 3.离线推理

1.将得到om模型后进行模型性能推理，在310上运行，先执行npu-smi info查看设备状态，确保device空闲

```
cd test
bsah om_inference.sh #产生文件在 result/dumpOutput_device0 
```

2.进行模型精度统计

eval_tools文件夹内要含有 ground_truth相关文件

```
cd eval_tools
python dsfd_acc_eval.py -p '../result/dumpOutput_device0/' -g './ground_truth/'
```

3.模型推理性能及精度

| Model | Batch Size | 310 (FPS/Card) | T4 (FPS/Card) | 310/T4    |
| ----- | ---------- | -------------- | ------------- | --------- |
| DSFD  | 1          | *206*          | *168*         | *206/168* |
| DSFD  | 4          | *262*          | *314*         | *262/314* |
| DSFD  | 8          | *286*          | *380*         | *286/380* |
| DSFD  | 16         | *306*          | *425*         | *306/425* |
| DSFD  | 32         | *305*          | *427*         | *305/427* |



| Framework | Atlas  NPU Model | Server          | Container | Precision | Dataset    | Accuracy                                                     | Ascend  AI Processor | NPU  Version         |
| --------- | ---------------- | --------------- | --------- | --------- | ---------- | ------------------------------------------------------------ | -------------------- | -------------------- |
| PyTorch   | Atlas 300-3010   | Atlas  800-3010 | NA        | fp16      | WIDER FACE | Easy  Val AP: 0.9443  Medium Val AP: 0.9347  Hard  Val AP: 0.8645 | Ascend  310          | Atlas  300-3010-32GB |