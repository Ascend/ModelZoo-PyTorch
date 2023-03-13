## 文件作用说明：

```
├── resnet34_aipp.config             //aipp工具数据集预处理配置文件
├── get_info.py                      //生成推理输入的数据集二进制info文件或jpg info文件
├── pth2onnx.py                      //用于转换pth模型文件到onnx模型文件
├── pytorch_transfer.py               //数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件
├── ReadMe.md
├── resnet34_atc.sh                  //onnx模型转换om模型脚本
├── val_label.txt                    //ImageNet数据集标签，用于验证推理结果
└── vision_metric_ImageNet.py        //验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy
```



## 推理端到端步骤：

1.  数据预处理，把ImageNet 50000张图片转为二进制文件（.bin）

   ```shell
   python3.7 pytorch_transfer.py resnet /home/HwHiAiUser/dataset/ImageNet/ILSVRC2012_img_val ./prep_bin
   ```

2. 生成数据集info文件

   ```shell
   python3.7 get_info.py bin ./prep_bin ./BinaryImageNet.info 256 256
   ```

3. 从torchvision下载[resnet34模型](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/ResNet34/PTH/resnet34-b627a593.pth)或者指定自己训练好的pth文件路径，通过pth2onnx.py脚本转化为onnx模型

   ```shell
   python3.7 pth2onnx.py ./resnet34-333f7ec4.pth ./resnet34_dynamic.onnx
   ```

4. 支持脚本将.onnx文件转为离线推理模型文件.om文件

   ${chip_name}可通过`npu-smi info`指令查看
   
    ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
   ```shell
   bash resnet34_atc.sh Ascend${chip_name} # Ascend310P3
   ```

5. 使用Benchmark工具进行推理
   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ./benchmark -model_type=vision -om_path=resnet34_fp16_bs16.om -device_id=0 -batch_size=16 -input_text_path=BinaryImageNet.info -input_width=256 -input_height=256 -useDvpp=false -output_binary=false
   ```

6. 精度验证，调用vision_metric_ImageNet.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中

   ```shell
   python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
   ```
7. 模型量化：
   a.生成量化数据：
   '''shell
      mkdir amct_prep_bin
      python3.7 pytorch_transfer_amct.py  /home/HwHiAiUser/dataset/ImageNet/ILSVRC2012_img_val ./amct_prep_bin
      mkdir data_bs64
      python3.7 calibration_bin ./amct_prep_bin data_bs64 64
   b. 量化模型转换：
      amct_onnx calibration --model resnet34_dynamic.onnx --save_path ./result/resnet34 --input_shape="actual_input_1:64,3,224,224" --data_dir "./data_bs64/" --data_type "float32"