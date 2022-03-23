export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${install_path}/atc/bin:${install_path}/bin:${install_path}/atc/ccec_compiler/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/lib64:${install_path}/atc/lib64:${install_path}/acllib/lib64:${install_path}/compiler/lib64/plugin/opskernel:${install_path}/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/latest/python/site-packages:${install_path}/opp/op_impl/built-in/ai_core/tbe:${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export ASCEND_AICPU_PATH=${install_path}
export ASCEND_OPP_PATH=${install_path}/opp
export TOOLCHAIN_HOME=${install_path}/toolkit
export ASCEND_AUTOML_PATH=${install_path}/tools
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/:${LD_LIBRARY_PATH}
atc --model=decoder_final.onnx --framework=5 --output=decoder_final --input_format=ND \
 --input_shape_range="memory:[10,1~1500,256];memory_mask:[10,1,1~1500];ys_in_pad:[10,1~1500];ys_in_lens:[10];r_ys_in_pad:[10,1~1500]" --out_nodes="Add_488:0;Add_977:0"  --log=error --soc_version=Ascend310


