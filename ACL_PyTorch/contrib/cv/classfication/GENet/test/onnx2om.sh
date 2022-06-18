source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --model=genet_gpu.onnx --framework=5 --input_format=NCHW --input_shape="image:1,3,32,32" --output=genet_bs1_tuned --soc_version=$1 --auto_tune_mode="GA" --log=debug
atc --model=genet_gpu.onnx --framework=5 --input_format=NCHW --input_shape="image:16,3,32,32" --output=genet_bs16_tuned --soc_version=$1 --auto_tune_mode="GA" --log=debug