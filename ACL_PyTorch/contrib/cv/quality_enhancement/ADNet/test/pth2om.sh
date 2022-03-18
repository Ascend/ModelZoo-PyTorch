python3.7.5 ADNet_pth2onnx.py model_70.pth ADNet.onnx
source env.sh
atc --framework=5 --model=ADNet.onnx --output=ADNet_bs1 --input_format=NCHW --input_shape="image:1,1,321,481" --log=debug --soc_version=Ascend310 
atc --framework=5 --model=ADNet.onnx --output=ADNet_bs4 --input_format=NCHW --input_shape="image:4,1,321,481" --log=debug --soc_version=Ascend310 
atc --framework=5 --model=ADNet.onnx --output=ADNet_bs8 --input_format=NCHW --input_shape="image:8,1,321,481" --log=debug --soc_version=Ascend310 
atc --framework=5 --model=ADNet.onnx --output=ADNet_bs16 --input_format=NCHW --input_shape="image:16,1,321,481" --log=debug --soc_version=Ascend310 
atc --framework=5 --model=ADNet.onnx --output=ADNet_bs32 --input_format=NCHW --input_shape="image:32,1,321,481" --log=debug --soc_version=Ascend310 