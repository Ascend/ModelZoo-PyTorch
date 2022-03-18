文件作用说明：

- Pth转换om脚本，pth转换om脚本
- ATC转换脚本atc_crnn.sh
- benchmark工具源码地址：https://gitee.com/ascend/cann-benchmark/tree/master/infer
- ONNX模型lstm算子修改脚本lstm_revise.py
- 测试数据集生成脚本parse_testdata.py
- 推理后处理脚本postpossess_CRNN_pytorch.py
- ReadMe.md
- 

推理端到端步骤：

（1） 使用脚本pth2onnx.py将pth文件导出为onnx文件



（2）运行atc_crnn.sh脚本转换om模型

本demo已提供调优完成的om模型



（3）用parse_testdata.py脚本处理数据集



（4）./benchmark.x86_64 -model_type=vision -batch_size=16 -device_id=0 -input_text_path=./crnn.info -input_width=100 -input_height=32 -om_path=./crnn_sim.om -useDvpp=False

运行benchmark推理，结果保存在 ./result 目录下



（5）python3.7 postpossess_CRNN_pytorch.py

验证推理结果

