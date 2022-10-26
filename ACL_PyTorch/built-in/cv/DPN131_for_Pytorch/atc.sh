source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --model=./dpn131.onnx --framework=5 --output=dpn131_fp16_bs8 --input_format=NCHW --input_shape="image:8,3,224,224" --log=error --soc_version=$1
