:<<!
ATC Parameters:
--input_format=ND # any format
!
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model='bmn-bs1.onnx' --output='bmn-bs1' --input_format=NCHW --input_shape="image:1,400,100" --log=debug --soc_version=Ascend310
