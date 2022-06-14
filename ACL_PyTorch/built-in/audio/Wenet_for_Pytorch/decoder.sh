export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

atc --model=decoder_final.onnx --framework=5 --output=decoder_final --input_format=ND \
 --input_shape_range="memory:[10,1~1500,256];memory_mask:[10,1,1~1500];ys_in_pad:[10,1~1500];ys_in_lens:[10];r_ys_in_pad:[10,1~1500]" --out_nodes="Add_488:0;Add_977:0"  --log=error --soc_version=$1


