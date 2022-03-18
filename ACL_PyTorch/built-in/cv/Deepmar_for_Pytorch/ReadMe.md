文件作用说明：

1.auto_tune.sh：模型转换脚本，集成了auto tune功能，可以手动关闭

2.pthtar2onnx.py：用于转换pth.tar文件到onnx文件

3.deepmar.info：PETA数据集信息，用于benchmark推理获取数据集

4.preprocess_deepmar_pytorch.py：数据集预处理脚本，通过均值方差处理归一化图片

5.label.json：PETA数据集标签，用于验证推理结果

6.postprocess_deepmar_pytorch.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy

7.benchmark工具源码地址：https://gitee.com/ascend/cann-benchmark/tree/master/infer





推理端到端步骤：

（1） 从开源仓https://github.com/dangweili/pedestrian-attribute-recognition-pytorch/blob/master/baseline/model/DeepMAR.py下载deepamar模型或者指定自己训练好的pth文件路径，使用提供的DeepMar.py替换掉模型中的DeepMar.py, 通过export_onnx.py脚本转化为onnx模型



（2）为提高性能，可以使用remove_pad.py剔除掉pad算子，运行auto_tune.sh脚本转换om模型，也可以选择手动关闭auto_tune，由于提出pad算子后，性能已经较好，本包中提供的om为未经auto_tune调优的om模型


（3）运行python script/dataset/transform_peta.py得到数据集，python split_test_data.py得到测试集txt信息image.txt和标签json文件label.json


（4）运行python preprocess_deepmar_pytorch.py dataset/peta/images input_bin image.txt，根据测试集image.txt生成对应bin文件

（5）运行python get_info.py input_bin deepmar.info 224 224 生成deepmar.info文件，存储bin文件的信息

（6）./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -om_path=deepmar_bs1.om -input_width=224 -input_height=224 -input_text_path=deepmar.info -useDvpp=false -output_binary=true

运行benchmark推理，结果保存在 ./result/dumpOutput_device0下 目录下



（7）python postprocess_deepmar_pytorch.py result/dumpOutput_device0/ label.json

验证推理结果，第一项即为acc

