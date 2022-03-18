文件作用说明：

1.atc.sh：模型转换脚本，生成动态分档模型

2.pth2onnx.py：用于转换ckpt文件到onnx文件

3.acl_net.py: pyACL推理依赖模块

4.interfaces.py: 替换speechbrain/pretrained 目录下同名文件

5.om_infer.sh: pyACL推理启动脚本

6.pyacl_infer.py: pyACL推理代码

7.tdnn_postprocess.py: 预处理脚本

8.tdnn_preprocess.py: 后处理脚本



推理端到端步骤：

（1） 从Speechbrain克隆源代码，修改speechbrain/nnet/CNN.py 349行padding_mode='constant',从Ascend Modelzoo获取训练好的权重文件夹best model, 进templates/speaker_id 运行 pth2onnx.py脚本生成tdnn.onnx模型

（2） 准备数据集，注释掉speechbrain/templates/speaker_id/mini_librispeech_prepare.py 174行代码，然后执行预处理脚本，python3 tdnn_preprocess.py, 将数据集处理为二进制文件

（3） 执行atc脚本, bash atc.sh tdnn.onnx tdnn，生成tdnn.om

（4） 执行om_infer脚本， bash om_infer.sh，推理结果输出在result目录下

 (5)  执行后处理脚本，python3 tdnn_postprocess.py 得到模型精度