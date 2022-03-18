文件作用说明：

1.auto_tune.sh：模型转换脚本，集成了auto tune功能，可以手动关闭

2.pth2onnx.py：用于转换pth文件到onnx文件

3.pthtar2onnx.py：用于转换pth.tar文件到onnx文件

4.BinaryImageNet.info：ImageNet数据集信息，用于benchmark推理获取数据集

5.PytorchTransfer.py：数据集预处理脚本，通过均值方差处理归一化图片

6.val_label.txt：ImageNet数据集标签，用于验证推理结果

7.vision_metric_ImageNet.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy

8.benchmark工具源码地址：https://gitee.com/ascend/cann-benchmark/tree/master/infer





推理端到端步骤：

（1） 从Torchvision下载mobilenetV2模型或者指定自己训练好的pth文件路径，通过pth2onnx.py脚本转化为onnx模型



（2）运行auto_tune.sh脚本转换om模型，也可以选择手动关闭auto_tune

本demo已提供调优完成的om模型



（3）用PytorchTransfer.py脚本处理数据集，参考BinaryImageNet.Info配置处理后的二进制数据集路径



（4）./benchmark.x86_64 -model_type=vision -batch_size=16 -device_id=0 -input_text_path=./BinaryImageNet.info -input_width=224 -input_height=224 -om_path=./resnet50_pytorch.om -useDvpp=False

运行benchmark推理，结果保存在 ./result 目录下



（5）python3.7 vision_metric_ImageNet.py result/dumpOutput/ ./val_label.txt ./ result.json

验证推理结果

