export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:$PATH
export PATH=${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export ASCEND_OPP_PATH=${install_path}/opp
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/acllib/lib64:$LD_LIBRARY_PATH

atc --model=encoder_revise.onnx --framework=5 --output=encoder --input_format=ND \
--input_shape_range="input:[1~1500,83]" --log=error --op_select_implmode=high_performance \
--soc_version=$1
