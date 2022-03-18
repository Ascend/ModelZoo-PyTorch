## 模型功能

 基于该模型实现图像分类。

## 原始模型


原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/classification/googlenet.onnx

AIPP下载地址 ：

https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/googlenet_onnx_picture/insert_op.cfg

## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/classification/googlenet_yuv.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --output_type=FP32 --input_shape="data:1,3,224,224" --input_format=NCHW --output="googlenet_yuv" --soc_version=Ascend310 --insert_op_conf=insert_op.cfg --framework=5  --model="/home/ascend/models/googlenet.onnx" 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，发起推理性能测试。可以参考如下指令： 

```
./msame --model ../model/googlenet_yuv.om --output output/ --loop 100
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 2.245480 ms
Inference average time without first time: 2.243010 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape:  224 *224 *3，不带AIPP，平均推理性能 2.24ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0305/155949_2c8f8d0c_8189215.jpeg "verify_test1.jpg")
