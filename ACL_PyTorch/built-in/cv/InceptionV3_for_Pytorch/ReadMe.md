# InceptionV3 模型离线推理指导

---

## 文件作用说明：

1. inceptionv3_pth2onnx.py：用于转换pth模型文件到onnx模型文件
2. gen_calibration_bin.py：生成bin格式数据集脚本，数据集用于量化校准
3. inceptionv3_atc.sh：onnx模型转换om模型脚本
4. imagenet_torch_preprocess.py：数据集预处理脚本，对图片进行缩放裁剪，生成图片二进制文件
5. aipp_inceptionv3_pth.config：数据集aipp预处理配置文件
6. ais_infer工具源码地址：https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer
7. vision_metric_ImageNet.py：验证推理结果脚本，比对ais_infer输出的分类结果和标签，给出Accuracy



## 推理端到端步骤：

1. 从 Torchvision 下载 PyTorch 预训练的 inceptionv3 模型（这里为 inceptionv3.pth），通过 inceptionv3_pth2onnx.py 脚本转化为onnx模型

   ```
   python3.7 inceptionv3_pth2onnx.py inceptionv3.pth inceptionv3.onnx
   ```

2. ONNX模型量化（可选）

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

   ${chip_name}可通过`npu-smi info`指令查看
   
   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
   
   ```
   bash inceptionv3_atc.sh Ascend${chip_name} # Ascend310P3
   ```

4. 用imagenet_torch_preprocess.py脚本处理数据集   

   ```
   python3.7 imagenet_torch_preprocess.py inceptionv3 /root/dataset/ImageNet/val_union ./prep_dataset
   ```

5. 设置环境变量   

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

6. 使用ais_infer离线推理

   按照 [ais_infer](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer) 文档，将 ais_infer 目录复制到当前工作目录下，编译安装 whl 文件。
   
   ```
   python3.7 ./ais_infer/ais_infer.py --model inceptionv3_bs8.om --device 0 --batchsize 8 --input ./prep_dataset --output ./result/ --outfmt TXT 
   ```
   
   运行 ais_infer 进行推理，结果会保存在 ./result/{timestamp} 目录下, 其中 {timestamp} 为运行推理工具的时间戳。
   
   目录下保存了每个图像的有推理结果和性能测试结果 summary.json 文件

7. 验证推理结果

   首先将 ./result/{timestamp} 目录下的 summary.json 文件移动到其他目录下，然后使用 vision_metric_ImageNet.py 进行推理结果验证，获得 Top1-Top5 Accuracy。
   
   ```
   python3.7 vision_metric_ImageNet.py  result/{timestamp}/  /root/dataset/ImageNet/val_label.txt  ./  result.json
   ```
   
   注意将 {timestamp} 更换为具体路径
   
   参数说明：

   - result/{timestamp}/：ais_infer 工具推理结果保存目录
   - /root/dataset/ImageNet/val_label.txt：ImageNet验证集标签文件
   - ./：验证结果的保存目录
   - result.json：验证结果保存文件名

## 模型获取

可以使用如下命令获取PyTorch框架的原始模型和转换后的Onnx模型

Pytorch：
```
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/InceptionV3/inception_v3.pth
```
ONNX：
```
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/InceptionV3/inceptionv3.onnx
```
