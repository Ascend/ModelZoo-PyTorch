文件作用说明：

1. inceptionv3_pth2onnx.py：用于转换pth模型文件到onnx模型文件
2. gen_calibration_bin.py：生成bin格式数据集脚本，数据集用于量化校准
3. inceptionv3_atc.sh：onnx模型转换om模型脚本
4. imagenet_torch_preprocess.py：数据集预处理脚本，对图片进行缩放裁剪，生成图片二进制文件
5. aipp_inceptionv3_pth.config：数据集aipp预处理配置文件
6. gen_dataset_info.py：生成推理输入的数据集二进制info文件
7. env.sh：环境变量文件
8. benchmark工具源码地址：https://gitee.com/ascend/cann-benchmark/tree/master/infer
9. vision_metric_ImageNet.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy



推理端到端步骤：

1. 从Torchvision下载inceptionv3模型，通过inceptionv3_pth2onnx.py脚本转化为onnx模型

2. ONNX模型量化

   1. AMCT工具包安装，具体参考《[CANN 开发辅助工具指南 01](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》中的昇腾模型压缩工具使用指南（ONNX）章节；

   2. 生成bin格式数据集，数据集用于校正量化因子。当前模型为动态batch，建议使用较大的batch size：

   ```
   python3.7 gen_calibration_bin.py inceptionv3 /root/dataset/ImageNet/val_union ./calibration_bin 32 1
   ```

   参数说明：

   - inceptionv3：模型类型
   - /root/dataset/ImageNet/val_union ：模型使用的数据集路径；
   - ./calibration_bin：生成的bin格式数据集路径；
   - 32：batch size；
   - 1：batch num。

   3. ONNX模型量化

   ```
   amct_onnx calibration --model inceptionv3.onnx  --save_path ./result/inceptionv3  --input_shape "actual_input_1:32,3,299,299" --data_dir "./calibration_bin" --data_types "float32" 
   ```

   会在result目录下生成inceptionv3_deploy_model.onnx量化模型

   4. 量化模型后续的推理验证流程和非量化一致。

3. 运行inceptionv3_atc.sh脚本转换om模型

4. 用imagenet_torch_preprocess.py脚本处理数据集   

```
python3.7 imagenet_torch_preprocess.py inceptionv3 /root/dataset/ImageNet/val_union ./prep_dataset
```

5. 生成推理输入的数据集二进制info文件     

```
python3.7 gen_dataset_info.py bin ./prep_dataset ./inceptionv3_prep_bin.info 299 299
```

6. 设置环境变量   

```
source env.sh
```

7. 使用benchmark离线推理

```
./benchmark.x86_64 -model_type=vision -om_path=inceptionv3_bs8.om -device_id=0 -batch_size=8 -input_text_path=inceptionv3_prep_bin.info -input_width=299 -input_height=299 -output_binary=False -useDvpp=False
```

运行benchmark推理，结果保存在 ./result 目录下

8. 验证推理结果

```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ /root/dataset/ImageNet/val_label.txt ./ result.json
```




模型获取

可以使用如下命令获取PyTorch框架的原始模型和转换后的Onnx模型

Pytorch：
```
wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/InceptionV3/inception_v3.pth
```
ONNX：
```
wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/InceptionV3/inceptionv3.onnx
```
