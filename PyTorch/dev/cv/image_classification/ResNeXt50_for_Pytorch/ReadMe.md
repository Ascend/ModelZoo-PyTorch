文件作用说明：

1.resnext50_pth2onnx.py：用于转换pth模型文件到onnx模型文件

2.resnext50_atc.sh：onnx模型转换om模型脚本

3.preprocess_resnext50_pth.py：数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件

4.aipp_resnext50_pth.config：数据集aipp预处理配置文件

5.get_info.py：生成推理输入的数据集二进制info文件或jpg info文件

6.benchmark：离线推理工具

7.vision_metric_ImageNet.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy



推理端到端步骤：

（1） 从Torchvision下载resnext50模型，通过resnext50_pth2onnx.py脚本转化为onnx模型

	python3.7 resnext50_pth2onnx.py ./resnext50_32x4d-7cdf4587.pth ./resnext50.onnx

（2）运行resnext50_atc.sh脚本转换om模型
	bash resnext50_atc.sh

本demo已提供调优完成的om模型



（3）用preprocess_resnext50_pth.py脚本处理数据集，参考resnext50_val.info配置处理后的二进制数据集路径。或者配置数据集aipp预处理文件aipp_resnext50_pth.config。
    python3 preprocess_resnext50_pth.py dataset/ImageNet/val_union/ pre_bin



（4）生成推理输入的数据集二进制info文件
    python3 get_info.py bin pre_bin resnext50_val.info 224 224



（5）增加benchmark.{arch}可执行权限。
	chmod u+x benchmark.x86_64
	
（6）使用benchmark离线推理
	./benchmark -model_type=vision -om_path=resnext50_bs16.om -device_id=0 -batch_size=16 -input_text_path=resnext50_val.info -input_width=224 -input_height=224 -useDvpp=false
	
	
（7）验证推理结果，运行benchmark推理，结果保存在 ./result 目录下
	python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
	


