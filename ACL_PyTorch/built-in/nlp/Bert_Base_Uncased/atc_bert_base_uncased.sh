export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATHexport ASCEND_OPP_PATH=${install_path}/opp

atc --input_format=ND --framework=5 --model=bert_base_batch_8_sim.onnx\
 --input_shape="input_ids:8,512;token_type_ids:8,512;attention_mask:8,512"\
 --output=bert_base_batch_8_sim_auto --auto_tune_mode="RL,GA"\
 --log=error --soc_version=Ascend710 --optypelist_for_implmode="Gelu"\
 --op_select_implmode=high_performance --input_fp16_nodes="attention_mask"