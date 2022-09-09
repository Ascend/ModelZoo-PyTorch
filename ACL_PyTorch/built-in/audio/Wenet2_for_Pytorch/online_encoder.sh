export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:$PATH
export PATH=${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export ASCEND_OPP_PATH=${install_path}/opp
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/acllib/lib64:$LD_LIBRARY_PATH

atc --model=online_encoder.onnx --framework=5 --output=online_encoder --input_format=ND --input_shape="chunk_xs:64,67,80;chunk_lens:64;offset:64;att_cache:64,12,4,64,128;cnn_cache:64,12,256,7;cache_mask:64,1,64" --log=error  --soc_version=$1

