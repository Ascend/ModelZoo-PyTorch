source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

export ASCEND_GLOBAL_LOG_LEVEL=3
for i in $(seq 0 7); do /usr/local/Ascend/driver/tools/msnpureport -g error -d $i; done

atc --input_format=ND --framework=5 --model=bert_base_batch_8.onnx\
 --input_shape="input_ids:8,512;token_type_ids:8,512;attention_mask:8,512"\
 --output=bert_base_batch_8_auto\
 --log=error --soc_version=$1 --optypelist_for_implmode="Gelu"\
 --op_select_implmode=high_performance --input_fp16_nodes="attention_mask"