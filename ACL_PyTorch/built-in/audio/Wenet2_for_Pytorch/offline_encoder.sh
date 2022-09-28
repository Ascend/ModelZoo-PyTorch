export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
atc --model=offline_encoder.onnx --framework=5 --output=offline_encoder --input_format=ND --input_shape_range="speech:[1~64,1~1500,80];speech_lengths:[1~64]" --log=error  --soc_version=$1

