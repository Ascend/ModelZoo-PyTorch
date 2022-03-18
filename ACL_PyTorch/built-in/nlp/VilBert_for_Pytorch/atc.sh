install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/toolkit/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export TOOLCHAIN_HOME=${install_path}/toolkit


atc --framework=5 --model=./models/vqa-vilbert_bs1_sim_modify.onnx --output=vqa-vilbert_bs1 --input_format=ND --input-shape="box-features:1,43,1024;box_coordinates:1,43,4;box_mask:1,43;q_token_ids:1,32;q_mask:1,32;q_type_ids:1,32" --out_nodes="Gemm_2971:0;Sigmoid_2972:0" --log=error --soc_version=Ascend310

