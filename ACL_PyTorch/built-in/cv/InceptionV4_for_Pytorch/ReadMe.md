文件作用说明：

1.inceptionv4_pth2onnx.py：用于转换pth模型文件到onnx模型文件

2.inceptionv4_atc.sh：onnx模型转换om模型脚本

3.preprocess_inceptionv4_pth.py：数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件

4.aipp_inceptionv4_pth.config：数据集aipp预处理配置文件

5.get_info.py：生成推理输入的数据集二进制info文件或jpg info文件

6.inceptionv4_val.info：ImageNet验证集二进制info文件，用于benchmark推理获取数据集

7.ImageNet.info：ImageNet验证集jpg info文件，用于benchmark推理获取数据集

8.val_label.txt：ImageNet数据集标签，用于验证推理结果

9.benchmark工具源码地址：https://gitee.com/ascend/cann-benchmark/tree/master/infer

10.vision_metric_ImageNet.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy





推理端到端步骤：

（1） 从https://github.com/Cadene/pretrained-models.pytorch下载inceptionv4模型，通过inceptionv4_pth2onnx.py脚本转化为onnx模型
    安装pretrainedmodels后需修改pretrained-models.pytorch/pretrainedmodels/models/inceptionv4.py：adaptiveAvgPoolWidth = features.shape[2].item()



（2）运行inceptionv4_atc.sh脚本转换om模型

本demo已提供调优完成的om模型



（3）用preprocess_inceptionv4_pth.py脚本处理数据集，参考inceptionv4_val.info配置处理后的二进制数据集路径。或者配置数据集aipp预处理文件aipp_inceptionv4_pth.config。
    python3 preprocess_inceptionv4_pth.py dataset/ImageNet/val_union/ prep_bin



（4）生成推理输入的数据集二进制info文件或jpg info文件
     python3 get_info.py bin prep_bin inceptionv4_val.info 299 299
     python3 get_info.py jpg dataset/ImageNet/val_union ImageNet.info



（5）使用benchmark离线推理
    ./benchmark -model_type=vision -om_path=inceptionv4_bs16.om -device_id=0 -batch_size=16 -input_text_path=inceptionv4_val.info -input_width=299 -input_height=299 -useDvpp=false
	或者
    ./benchmark -model_type=vision -om_path=inceptionv4_bs1.om -device_id=0 -batch_size=1 -input_text_path=ImageNet.info -input_width=336 -input_height=336 -useDvpp=true

运行benchmark推理，结果保存在 ./result 目录下



（6）python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json

验证推理结果

