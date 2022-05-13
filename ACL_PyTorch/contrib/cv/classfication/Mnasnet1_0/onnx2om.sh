atc --framework=5 --model=mnasnet1.0.onnx --output=mnasnet1.0_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend710

atc --framework=5 --model=mnasnet1.0.onnx --output=mnasnet1.0_bs4 --input_format=NCHW --input_shape="image:4,3,224,224" --log=debug --soc_version=Ascend710

atc --framework=5 --model=mnasnet1.0.onnx --output=mnasnet1.0_bs8 --input_format=NCHW --input_shape="image:8,3,224,224" --log=debug --soc_version=Ascend710

atc --framework=5 --model=mnasnet1.0.onnx --output=mnasnet1.0_bs16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend710

atc --framework=5 --model=mnasnet1.0.onnx --output=mnasnet1.0_bs32 --input_format=NCHW --input_shape="image:32,3,224,224" --log=debug --soc_version=Ascend710

atc --framework=5 --model=mnasnet1.0.onnx --output=mnasnet1.0_bs64 --input_format=NCHW --input_shape="image:64,3,224,224" --log=debug --soc_version=Ascend710