文件作用说明：

1.auto_tune.sh：模型转换脚本，集成了auto tune功能，可以手动关闭

2.pth2onnx.py：用于转换pth文件到onnx文件

3.pthtar2onnx.py：用于转换pth.tar文件到onnx文件

4.BinaryImageNet.info：ImageNet数据集信息，用于benchmark推理获取数据集

5.PytorchTransfer.py：数据集预处理脚本，通过均值方差处理归一化图片

6.val_label.txt：ImageNet数据集标签，用于验证推理结果

7.vision_metric_ImageNet.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy

8.benchmark工具源码地址：https://gitee.com/ascend/cann-benchmark/tree/master/infer

710增加文件说明：

1.pthtar2onx_dynamic.py：用于转换pth.tar文件到动态onnx文件；

2.imagenet_torch_preprocess.py：imagenet数据集预处理；

3.gen_dataset_info.py：获取imagenet数据集信息info文件脚本；

4.gen_resnet50_64bs_bin.py：基于数据预处理结果合成量化所需的64bs输入；

5.aipp_resnet50_710.aippconfig：aipp配置文件

推理端到端步骤：

（1） 从Torchvision下载resnet50模型或者指定自己训练好的pth文件路径，通过pth2onnx.py脚本转化为onnx模型



（2）运行auto_tune.sh脚本转换om模型，也可以选择手动关闭auto_tune

本demo已提供调优完成的om模型



（3）用PytorchTransfer.py脚本处理数据集，参考BinaryImageNet.Info配置处理后的二进制数据集路径



（4）./benchmark.x86_64 -model_type=vision -batch_size=16 -device_id=0 -input_text_path=./BinaryImageNet.info -input_width=224 -input_height=224 -om_path=./resnet50_pytorch.om -useDvpp=False

运行benchmark推理，结果保存在 ./result 目录下



（5）python3.7 vision_metric_ImageNet.py result/dumpOutput/ ./val_label.txt ./ result.json

验证推理结果

710精度验证步骤：

（1）python3.7.5 imagenet_torch_preprocess.py resnet ./ImageNet/val_union ./prep_dataset
数据集处理；

（2）python3.7.5 gen_dataset_info.py ./prep_dataset ./resnet50_prep_bin.info 256 256
获取数据集信息info文件；

（3）./benchmark.x86_64 -model_type=vision -batch_size=16 -device_id=0 -input_text_path=./resnet50_prep_bin.info -input_width=256 -input_height=256 -om_path=./resnet50_pytorch.om -output_binary=False -useDvpp=False
数据集信息输入，调用benchmark完成推理OM模型；

（4）python3.7.5 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result_prep.json
查看result_prep.json中精度结果。

