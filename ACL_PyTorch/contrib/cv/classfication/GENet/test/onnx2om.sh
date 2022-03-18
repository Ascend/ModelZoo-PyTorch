export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

atc --model=genet_gpu.onnx --framework=5 --input_format=NCHW --input_shape="image:1,3,32,32" --output=genet_bs1_tuned --soc_version=Ascend310 --auto_tune_mode="GA" --log=debug
atc --model=genet_gpu.onnx --framework=5 --input_format=NCHW --input_shape="image:16,3,32,32" --output=genet_bs16_tuned --soc_version=Ascend310 --auto_tune_mode="GA" --log=debug