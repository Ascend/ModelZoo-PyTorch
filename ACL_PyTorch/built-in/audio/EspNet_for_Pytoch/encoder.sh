export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:$PATH
export PATH=${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export ASCEND_OPP_PATH=${install_path}/opp
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/acllib/lib64:$LD_LIBRARY_PATH

atc --model=encoder_revise.onnx --framework=5 --output=encoder_262_1478 --input_format=ND \
--input_shape="input:-1,83" --log=error --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_performance \
--dynamic_dims="262;326;390;454;518;582;646;710;774;838;902;966;1028;1284;1478" \
--soc_version=$1
