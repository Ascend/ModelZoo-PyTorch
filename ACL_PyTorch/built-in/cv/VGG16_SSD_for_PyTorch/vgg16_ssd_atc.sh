source /usr/local/Ascend/ascend-toolkit/set_env.sh
export REPEAT_TUNE=True

atc --model=./vgg16_ssd.onnx --framework=5 --output=vgg16_ssd --input_format=NCHW --input_shape="actual_input_1:1,3,300,300" --log=info --soc_version=$1

atc --model= ./result_amct/vgg16_ssd_deploy_model.onnx --framework=5 --output=vgg16_ssd_deploy_model --input_format=NCHW --input_shape="actual_input_1:1,3,300,300" --log=info --soc_version=$1